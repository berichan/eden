// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <stop_token>
#include <vector>
#include "core/hle/service/nvdrv/core/native_gpu_channel.h"
#include "video_core/control/native_command_stream.h"

namespace Core {
class System;
}

namespace Service::Nvidia::NvCore {

// The submit count reserves increments already present in the guest stream.
// Only an executed, synchronized shim write may consume one reservation unit.
class NativeGpuFenceProgress {
public:
    NativeGpuFenceProgress(u32 id_, u32 target_, u32 requested_)
        : id{id_}, target{target_}, requested{requested_} {}
    bool HasRemaining() const {
        return requested <= target && executed < requested;
    }
    bool IsComplete() const {
        return requested <= target && executed == requested;
    }
    u32 GetExecuted() const {
        return executed;
    }
    bool CanWriteCounter(GPUVAddr owned_address, GPUVAddr address, size_t size) const {
        return HasRemaining() && size == sizeof(u32) && address == owned_address;
    }
    bool Signal(Tegra::Host1x::SyncpointManager& manager) {
        if (!HasRemaining() ||
            !manager.PublishNativeCompletion(id, target - requested + executed + 1, 1)) {
            return false;
        }
        ++executed;
        return true;
    }

private:
    const u32 id, target, requested;
    u32 executed{};
};

class NativeGpuSubmission {
public:
    // Complete retires this FIFO stream and drains issued work. Dependency-gated QMDs
    // remain in the retained graphics context until a later scheduling signal.
    enum class State { Reserved, Queued, Running, Complete, Failed, Cancelled };
    ~NativeGpuSubmission();
    bool MarkQueued();
    void Execute(Core::System& system, std::stop_token stop_token);
    void Cancel();
    // Only the producer may use this after an enqueue failed, before worker visibility.
    void CancelUnpublished();
    State GetState() const;
    Tegra::Control::NativeCommandStream::Result GetResult() const;
    const std::vector<u64>& GetEntries() const;

private:
    friend class NativeGpuChannel;
    NativeGpuSubmission(std::shared_ptr<NativeGpuChannel> channel, std::vector<u64> entries,
                        NvFence fence, u32 increments);
    void Release();
    void Finish(State state, Tegra::Control::NativeCommandStream::Result result);
    bool ReleaseSyncpoint(Core::System& system, u32 upper, u32 lower, u32 payload, u32 operation,
                          std::stop_token stop_token);

    const std::shared_ptr<NativeGpuChannel> channel;
    const std::shared_ptr<const NativeGpuChannel::CommandRing> ring;
    const std::shared_ptr<const Tegra::Control::NativeGpuGraphicsContext> graphics;
    const std::vector<u64> entries;
    const NvFence fence;
    NativeGpuFenceProgress fence_progress;
    const std::optional<u32> timeout;
    u64 sequence{};
    std::atomic<State> state{State::Reserved};
    std::atomic<bool> released{};
    mutable std::mutex result_mutex;
    Tegra::Control::NativeCommandStream::Result result;
};

} // namespace Service::Nvidia::NvCore
