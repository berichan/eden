// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/assert.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "common/thread.h"
#include "core/core.h"
#include "core/frontend/graphics_context.h"
#include "core/hle/service/nvdrv/core/native_gpu_submission.h"
#include "video_core/control/scheduler.h"
#include "video_core/dma_pusher.h"
#include "video_core/gpu.h"
#include "video_core/gpu_thread.h"
#include "video_core/host1x/host1x.h"
#include "video_core/renderer_base.h"

namespace VideoCommon::GPUThread {

ThreadManager::ThreadManager(Core::System& system_)
    : system{system_}
{}

ThreadManager::~ThreadManager() {
    NotifyShutdown();
}

void ThreadManager::StartThread(VideoCore::RendererBase& renderer, Core::Frontend::GraphicsContext& context, Tegra::Control::Scheduler& scheduler) {
    rasterizer = renderer.ReadRasterizer();
    {
        std::scoped_lock lock{state.write_lock};
        accepting_native = true;
    }
    thread = std::jthread([&](std::stop_token stop_token) {
        Common::SetCurrentThreadName("GPU");
        Common::SetCurrentThreadPriority(Common::ThreadPriority::Critical);
        Common::SetCurrentThreadToPerformanceCores();
        system.RegisterHostThread();

        auto current_context = context.Acquire();
        CommandDataContainer next;
        while (!stop_token.stop_requested()) {
            state.queue.PopWait(next, stop_token);
            if (stop_token.stop_requested()) {
                if (auto* native = std::get_if<SubmitNativeCommand>(&next.data)) {
                    native->submission->Cancel();
                }
                break;
            }
            if (auto* submit_list = std::get_if<SubmitListCommand>(&next.data)) {
                scheduler.Push(system.GPU(), submit_list->channel, std::move(submit_list->entries));
            } else if (auto* native = std::get_if<SubmitNativeCommand>(&next.data)) {
                native->submission->Execute(system, stop_token);
            } else if (std::holds_alternative<GPUTickCommand>(next.data)) {
                system.GPU().TickWork();
            } else if (const auto* flush = std::get_if<FlushRegionCommand>(&next.data)) {
                renderer.ReadRasterizer()->FlushRegion(flush->addr, flush->size);
            } else if (const auto* invalidate = std::get_if<InvalidateRegionCommand>(&next.data)) {
                renderer.ReadRasterizer()->OnCacheInvalidation(invalidate->addr, invalidate->size);
            } else {
                ASSERT(false);
            }
            DrainNativeRetirements();
            state.signaled_fence.store(next.fence);
            if (next.block) {
                // We have to lock the write_lock to ensure that the condition_variable wait not get a
                // race between the check and the lock itself.
                std::scoped_lock lk{state.write_lock};
                state.cv.notify_all();
            }
        }
    });
}

bool ThreadManager::SubmitNative(
    std::shared_ptr<Service::Nvidia::NvCore::NativeGpuSubmission> submission) {
    std::scoped_lock lock{state.write_lock};
    if (!accepting_native || !submission || !submission->MarkQueued()) {
        return false;
    }
    const u64 fence = state.last_fence + 1;
    if (!state.queue.TryEmplace(CommandData{SubmitNativeCommand{submission}}, fence, false)) {
        submission->CancelUnpublished();
        return false;
    }
    state.last_fence = fence;
    return true;
}

void ThreadManager::SubmitList(s32 channel, Tegra::CommandList&& entries, bool is_async) {
    PushCommand(SubmitListCommand(channel, std::move(entries)), false, is_async);
}

void ThreadManager::RetireNative(
    std::shared_ptr<Service::Nvidia::NvCore::NativeGpuChannel> channel) {
    std::scoped_lock lock{state.write_lock};
    if (!accepting_native) {
        return;
    }
    const u32 index = channel->GetChannelIndex();
    ASSERT(index < native_retirements.size());
    {
        std::scoped_lock retirement_lock{retirement_mutex};
        native_retirements[index] = std::move(channel);
    }
    const u64 fence = state.last_fence + 1;
    if (state.queue.TryEmplace(CommandData{GPUTickCommand{}}, fence, false)) {
        state.last_fence = fence;
    }
}

void ThreadManager::DrainNativeRetirements() {
    decltype(native_retirements) retired;
    {
        std::scoped_lock lock{retirement_mutex};
        retired.swap(native_retirements);
    }
    for (const auto& channel : retired) {
        if (channel) {
            if (!channel->RetireBackend(system)) {
                std::scoped_lock lock{retirement_mutex};
                native_retirements[channel->GetChannelIndex()] = channel;
            }
        }
    }
}

void ThreadManager::FlushRegion(DAddr addr, u64 size, bool is_async) {
    if (!is_async) {
        // Always flush with synchronous GPU mode
        PushCommand(FlushRegionCommand(addr, size), false, is_async);
    }
}

void ThreadManager::TickGPU(bool is_async) {
    PushCommand(GPUTickCommand(), false, is_async);
}

void ThreadManager::InvalidateRegion(DAddr addr, u64 size) {
    rasterizer->OnCacheInvalidation(addr, size);
}

void ThreadManager::FlushAndInvalidateRegion(DAddr addr, u64 size, bool is_async) {
    if (Settings::IsGPULevelHigh()) {
        if (!is_async) {
            PushCommand(FlushRegionCommand(addr, size), false, is_async);
        } else {
            auto& gpu = system.GPU();
            const u64 fence = gpu.RequestFlush(addr, size);
            TickGPU(is_async);
            gpu.WaitForSyncOperation(fence);
        }
    }
    rasterizer->OnCacheInvalidation(addr, size);
}

u64 ThreadManager::PushCommand(CommandData&& command_data, bool block, bool is_async) {
    if (!is_async) {
        // In synchronous GPU mode, block the caller until the command has executed
        block = true;
    }

    std::unique_lock lk(state.write_lock);
    const u64 fence{++state.last_fence};
    state.queue.EmplaceWait(std::move(command_data), fence, block);

    if (block) {
        state.cv.wait(lk, thread.get_stop_token(), [this, fence] {
            return fence <= state.signaled_fence.load(std::memory_order_relaxed);
        });
    }

    return fence;
}

void ThreadManager::NotifyShutdown() {
    {
        std::scoped_lock lock{state.write_lock};
        accepting_native = false;
    }
    if (thread.joinable()) {
        thread.request_stop();
        thread.join();
    }
    CommandDataContainer pending;
    while (state.queue.TryPop(pending)) {
        if (auto* native = std::get_if<SubmitNativeCommand>(&pending.data)) {
            native->submission->Cancel();
        }
    }
}

} // namespace VideoCommon::GPUThread
