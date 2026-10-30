// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <thread>
#include "common/logging.h"
#include "common/scope_exit.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/hle/service/nvdrv/core/native_gpu_error_notifier.h"
#include "core/hle/service/nvdrv/core/native_gpu_submission.h"
#include "video_core/control/channel_state.h"
#include "video_core/control/native_atomic_sequence.h"
#include "video_core/control/native_gpu_context.h"
#include "video_core/control/native_report_semaphore.h"
#include "video_core/control/native_semaphore_acquire.h"
#include "video_core/control/native_semaphore_release.h"
#include "video_core/gpu.h"
#include "video_core/host1x/host1x.h"
#include "video_core/host1x/syncpoint_shim.h"
#include "video_core/memory_manager.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/renderer_base.h"

namespace Service::Nvidia::NvCore {
using Stream = Tegra::Control::NativeCommandStream;

struct NativeGpuChannel::Backend {
    std::shared_ptr<Tegra::Control::ChannelState> channel;
    std::array<u32, 8> classes{};
    std::array<u32, 0x40> puller{};
    Tegra::Control::NativeAtomicSequence atomic_sequence;

    bool PendingFrontendWork(const Tegra::Control::NativeGpuGraphicsContext& context) const {
        // Dependency-gated QMDs are parked context state, not an unfinished FIFO packet.
        // SignalQmd executes ready descriptors synchronously and retains deferred ones.
        return context.HasQmdFailure() ||
               channel->payload->maxwell_3d.HasPendingNativeInlineUpload() ||
               channel->payload->maxwell_3d.HasPendingNativeInstrumentation() ||
               channel->payload->maxwell_3d.HasPendingNativeMacro() ||
               channel->payload->maxwell_3d.HasNativeMacroFailure() ||
               channel->payload->kepler_compute.HasPendingNativeMacro() ||
               channel->payload->kepler_compute.HasPendingNativeInlineUpload() ||
               channel->payload->kepler_compute.HasNativeMethodFailure();
    }

    bool PendingFrontend(const Tegra::Control::NativeGpuGraphicsContext& context) const {
        return atomic_sequence.IsActive() || PendingFrontendWork(context);
    }

    bool Call(Core::System& system, const Tegra::Control::NativeGpuGraphicsContext& context,
              u32 subchannel, u32 method, u32 value, bool last, std::stop_token stop_token) {
        auto& gpu = system.GPU();
        auto& rasterizer = *gpu.Renderer().ReadRasterizer();
        auto& payload = *channel->payload;
        if (stop_token.stop_requested() || !atomic_sequence.AllowsSubchannel(subchannel)) {
            return false;
        }
        // This restricted scope supports state/ordering commands, not launches or
        // macros whose output could publish a release before the scope is closed.
        const bool restricted_scope =
            atomic_sequence.IsActive() || payload.maxwell_3d.HasPendingNativeInstrumentation();
        if ((classes[subchannel] == 0xc797 || classes[subchannel] == 0xc7c0) && method == 0x6c3 &&
            PendingFrontend(context)) {
            LOG_ERROR(HW_GPU, "Native report release blocked by unfinished frontend work");
            return false;
        }
        if (restricted_scope && (method == 0x6c || method == 0x6d)) {
            return false;
        }
        if ((restricted_scope || payload.maxwell_3d.HasPendingNativeInlineUpload() ||
             payload.kepler_compute.HasPendingNativeInlineUpload()) &&
            (method >= 0xe00 || (classes[subchannel] == 0xc7c0 && method == 0xb0))) {
            return false;
        }
        if (method == 0) {
            if ((value & ~0x1fffffU) || PendingFrontend(context)) {
                return false;
            }
            const u32 class_id = value & 0xffff;
            if (class_id != 0xc797 && class_id != 0xc7c0 && class_id != 0x902d &&
                class_id != 0xa140 && class_id != 0xc7b5) {
                return false;
            }
            classes[subchannel] = class_id;
            return true;
        }
        if (method < 0x40) {
            switch (method) {
            case 2: // NO_OPERATION
                return true;
            case 4: // SEMAPHOREA
                if (value > 0xff)
                    return false;
                break;
            case 5: // SEMAPHOREB
                if (value & 3)
                    return false;
                break;
            case 6:   // SEMAPHOREC
            case 0xa: // MEM_OP_A-C
            case 0xb:
            case 0xc:
                break;
            case 7: { // SEMAPHORED: checked acquire and release forms
                if (value == 0x100002) {
                    if (PendingFrontend(context))
                        return false;
                    using Release = Tegra::Control::NativeSemaphoreRelease;
                    struct ReleaseContext {
                        const Tegra::Control::NativeGpuGraphicsContext& graphics;
                        VideoCore::RasterizerInterface& rasterizer;
                        Core::System& system;
                        std::stop_token token;
                    } release_context{context, rasterizer, system, stop_token};
                    const Release::Host host{
                        &release_context,
                        [](void* opaque, GPUVAddr address, size_t size) {
                            return static_cast<ReleaseContext*>(opaque)
                                ->graphics.GetAddressSpace()
                                ->IsWritableRange(address, size);
                        },
                        [](void* opaque) {
                            return static_cast<ReleaseContext*>(opaque)
                                ->rasterizer.SynchronizeNativeWork();
                        },
                        [](void* opaque) {
                            return static_cast<u64>(static_cast<ReleaseContext*>(opaque)
                                                        ->system.CoreTiming()
                                                        .GetGlobalTimeNs()
                                                        .count());
                        },
                        [](void* opaque, GPUVAddr address, u32 payload, u64 timestamp) {
                            auto& self = *static_cast<ReleaseContext*>(opaque);
                            return self.graphics.GetAddressSpace()->WriteTimestampedReleaseChecked(
                                address, payload, timestamp, self.token);
                        },
                        [](void* opaque) {
                            return static_cast<ReleaseContext*>(opaque)->token.stop_requested();
                        },
                    };
                    const auto status =
                        Release::Release(puller[4], puller[5], puller[6], value, host);
                    if (status != Release::Error::None) {
                        LOG_ERROR(
                            HW_GPU,
                            "Native timestamped release failed: upper={:#x}, lower={:#x}, error={}",
                            puller[4], puller[5], static_cast<u32>(status));
                    }
                    return status == Release::Error::None;
                }
                if (value == 0x1004) {
                    if (PendingFrontend(context))
                        return false;
                    using Acquire = Tegra::Control::NativeSemaphoreAcquire;
                    struct AcquireContext {
                        const Tegra::Control::NativeGpuGraphicsContext& graphics;
                        VideoCore::RasterizerInterface& rasterizer;
                        std::stop_token token;
                    } acquire_context{context, rasterizer, stop_token};
                    const Acquire::Host host{
                        &acquire_context,
                        [](void* opaque, GPUVAddr address, size_t size) {
                            return static_cast<AcquireContext*>(opaque)
                                ->graphics.GetAddressSpace()
                                ->IsReadableRange(address, size);
                        },
                        [](void* opaque) {
                            return static_cast<AcquireContext*>(opaque)
                                ->rasterizer.SynchronizeNativeWork();
                        },
                        [](void* opaque, GPUVAddr address, u32& output) {
                            u32_le data{};
                            if (!static_cast<AcquireContext*>(opaque)
                                     ->graphics.GetAddressSpace()
                                     ->ReadBlockChecked(
                                         address, {reinterpret_cast<u8*>(&data), sizeof(data)}))
                                return false;
                            output = data;
                            return true;
                        },
                        [](void* opaque) {
                            return static_cast<AcquireContext*>(opaque)->token.stop_requested();
                        },
                    };
                    const auto status =
                        Acquire::Check(puller[4], puller[5], puller[6], value, host);
                    if (status != Acquire::Error::None) {
                        LOG_ERROR(HW_GPU,
                                  "Native semaphore acquire cannot continue: upper={:#x}, "
                                  "lower={:#x}, payload={:#x}, error={}",
                                  puller[4], puller[5], puller[6], static_cast<u32>(status));
                    }
                    return status == Acquire::Error::None;
                }
                if ((value & ~0x0110001fU) || (value & 0x1f) != 2 || !(value & (1U << 24)) ||
                    PendingFrontend(context) ||
                    (!(value & (1U << 20)) && context.HasPendingQmd())) {
                    return false;
                }
                const GPUVAddr address = (GPUVAddr{puller[4]} << 32) | puller[5];
                if (!context.GetAddressSpace()->IsWritableRange(address, sizeof(u32)) ||
                    !rasterizer.SynchronizeNativeWork() || stop_token.stop_requested()) {
                    return false;
                }
                const u32 data = puller[6];
                if (!context.GetAddressSpace()->WriteBlockChecked(
                        address, {reinterpret_cast<const u8*>(&data), sizeof(data)})) {
                    return false;
                }
                break;
            }
            case 0xd: { // C76F MEM_OP_D: serialize supported cache/memory barriers.
                const u32 operation = value >> 27;
                if ((value & 0x07ffffffU) ||
                    (operation != 5 && operation != 14 && operation != 16) ||
                    !rasterizer.SynchronizeNativeWork()) {
                    return false;
                }
                gpu.InvalidateGPUCache();
                break;
            }
            case 0x1e: // WFI
                if (value > 1 || !rasterizer.SynchronizeNativeWork())
                    return false;
                break;
            default:
                return false;
            }
            puller[method] = value;
            return true;
        }
        if ((classes[subchannel] == 0xc797 || classes[subchannel] == 0xc7c0) &&
            (method == 0x52 || method == 0x53)) {
            // The GPU worker cannot bind another channel during this job. Keep scopes
            // inside that execution boundary; pending frontend state rejects retirement.
            if (PendingFrontendWork(context)) {
                return false;
            }
            return method == 0x52 ? atomic_sequence.Begin(subchannel, value)
                                  : atomic_sequence.End(subchannel, value);
        }
        switch (classes[subchannel]) {
        case 0xc797:
            return payload.maxwell_3d.CallNativeMethod(system, method, value, last, stop_token);
        case 0xc7c0:
            return payload.kepler_compute.CallNativeMethod(method, value, last, stop_token);
        case 0x902d:
            // This class is unchanged, but executable 2D methods need validated native resources.
            if (method >= Tegra::Engines::Fermi2D::Regs::NUM_REGS ||
                payload.fermi_2d.execution_mask[method]) {
                return false;
            }
            payload.fermi_2d.CallMethod(system, method, value, last);
            return true;
        default:
            return false;
        }
    }
};

bool NativeGpuChannel::RetireBackend(Core::System& system) {
    std::shared_ptr<Backend> retired;
    {
        std::scoped_lock lock{mutex};
        if (open) {
            return true;
        }
        retired = std::move(backend);
    }
    if (retired) {
        try {
            system.GPU().ReleaseChannel(*retired->channel);
        } catch (const std::exception& error) {
            std::scoped_lock lock{mutex};
            backend = std::move(retired);
            LOG_ERROR(HW_GPU, "Native channel retirement deferred: {}", error.what());
            return false;
        }
    }
    return true;
}

NativeGpuSubmission::NativeGpuSubmission(std::shared_ptr<NativeGpuChannel> channel_,
                                         std::vector<u64> entries_, NvFence fence_, u32 increments_)
    : channel{std::move(channel_)}, ring{channel->command_ring},
      graphics{channel->graphics_context}, entries{std::move(entries_)}, fence{fence_},
      fence_progress{static_cast<u32>(fence_.id), fence_.value, increments_},
      timeout{channel->timeout} {}

NativeGpuSubmission::~NativeGpuSubmission() {
    Release();
}

void NativeGpuSubmission::Release() {
    if (sequence && !released.exchange(true)) {
        channel->ReleaseSubmission(sequence, state.load() == State::Complete);
    }
}

bool NativeGpuSubmission::MarkQueued() {
    std::scoped_lock lock{channel->mutex};
    const auto found = std::ranges::find(channel->pending_submissions, sequence,
                                         &NativeGpuChannel::PendingSubmission::sequence);
    if (found == channel->pending_submissions.end() || !channel->open || channel->faulted ||
        std::any_of(channel->pending_submissions.begin(), found,
                    [](const auto& job) { return !job.released && !job.published; })) {
        return false;
    }
    State expected = State::Reserved;
    if (!state.compare_exchange_strong(expected, State::Queued)) {
        return false;
    }
    found->published = true;
    return true;
}

void NativeGpuSubmission::Cancel() {
    auto expected = state.load();
    while (expected == State::Reserved || expected == State::Queued) {
        if (state.compare_exchange_weak(expected, State::Cancelled)) {
            Release();
            return;
        }
    }
}

void NativeGpuSubmission::CancelUnpublished() {
    {
        std::scoped_lock lock{channel->mutex};
        State expected = State::Queued;
        if (!state.compare_exchange_strong(expected, State::Cancelled)) {
            return;
        }
        const auto found = std::ranges::find(channel->pending_submissions, sequence,
                                             &NativeGpuChannel::PendingSubmission::sequence);
        ASSERT(found != channel->pending_submissions.end());
        found->published = false;
    }
    Release();
}

void NativeGpuSubmission::Finish(State final_state, Stream::Result final_result) {
    {
        std::scoped_lock lock{result_mutex};
        result = final_result;
    }
    state.store(final_state, std::memory_order_release);
    Release();
}

NativeGpuSubmission::State NativeGpuSubmission::GetState() const {
    return state.load(std::memory_order_acquire);
}

Stream::Result NativeGpuSubmission::GetResult() const {
    std::scoped_lock lock{result_mutex};
    return result;
}

const std::vector<u64>& NativeGpuSubmission::GetEntries() const {
    return entries;
}

bool NativeGpuSubmission::ReleaseSyncpoint(Core::System& system, u32 upper, u32 lower, u32 payload,
                                           u32 operation, std::stop_token stop_token,
                                           bool compute) {
    const auto base = graphics->GetAddressSpace()->GetSyncpointShimAddress();
    if (!base) {
        return false;
    }
    struct Host {
        NativeGpuSubmission& submission;
        Core::System& system;
        std::stop_token token;
        GPUVAddr address;
    } host{*this, system, stop_token,
           *base + u64{static_cast<u32>(fence.id)} * Tegra::Host1x::SyncpointShim::Stride};
    using Report = Tegra::Control::NativeReportSemaphore;
    const Report::Host bindings{
        &host,
        [](void* opaque, GPUVAddr address, size_t size) {
            const auto& self = *static_cast<Host*>(opaque);
            return self.submission.fence_progress.CanWriteCounter(self.address, address, size);
        },
        [](void* opaque) {
            auto& self = *static_cast<Host*>(opaque);
            return self.system.GPU().Renderer().ReadRasterizer()->SynchronizeNativeWork();
        },
        [](void* opaque, GPUVAddr, u32) {
            auto& self = *static_cast<Host*>(opaque);
            {
                std::scoped_lock lock{self.submission.channel->mutex};
                if (!self.submission.channel->open || self.submission.channel->faulted ||
                    self.token.stop_requested()) {
                    return false;
                }
            }
            // NVIDIA's shim increments once for any written payload. Notify only after
            // this actual stream position drained, with no channel lock held in callbacks.
            return self.submission.fence_progress.Signal(
                self.system.Host1x().GetSyncpointManager());
        },
        [](void* opaque) { return static_cast<Host*>(opaque)->token.stop_requested(); },
    };
    const auto release_result =
        Report::Release(upper, lower, payload, operation, bindings,
                        compute ? Report::Engine::Compute : Report::Engine::Graphics);
    if (release_result != Report::Error::None) {
        LOG_ERROR(
            HW_GPU,
            "Native syncpoint report rejected: upper={:#x}, lower={:#x}, expected={:#x}, error={}",
            upper, lower, host.address, static_cast<u32>(release_result));
    }
    return release_result == Report::Error::None;
}

void NativeGpuSubmission::Execute(Core::System& system, std::stop_token stop_token) {
    State expected = State::Queued;
    if (!state.compare_exchange_strong(expected, State::Running)) {
        return;
    }
    std::stop_source cancellation;
    std::stop_callback worker_stop{stop_token, [&] { cancellation.request_stop(); }};
    std::stop_callback channel_stop{channel->cancellation.get_token(),
                                    [&] { cancellation.request_stop(); }};
    std::atomic<bool> timed_out{};
    std::jthread watchdog;
    Stream::Result executed;
    bool activated_backend{};
    try {
        if (timeout && *timeout && !cancellation.stop_requested()) {
            watchdog = std::jthread([&](std::stop_token token) {
                std::mutex mutex;
                std::condition_variable_any condition;
                std::unique_lock lock{mutex};
                condition.wait_for(lock, token, std::chrono::milliseconds{*timeout},
                                   [] { return false; });
                if (!token.stop_requested()) {
                    timed_out.store(true);
                    cancellation.request_stop();
                }
            });
        }
        if (cancellation.stop_requested()) {
            Finish(State::Cancelled, {Stream::Error::Cancelled});
            return;
        }
        const auto token = cancellation.get_token();
        auto& gpu = system.GPU();
        if (!channel->backend) {
            auto backend = std::make_shared<NativeGpuChannel::Backend>();
            backend->channel = gpu.AllocateChannel();
            bool published{};
            SCOPE_EXIT {
                if (!published) {
                    try {
                        gpu.ReleaseChannel(*backend->channel);
                    } catch (const std::exception&) {
                        // Keep partial registration reachable for retirement after a failed
                        // cleanup.
                        std::scoped_lock lock{channel->mutex};
                        channel->backend = backend;
                    }
                }
            };
            backend->channel->native_context = graphics;
            backend->channel->memory_manager = graphics->GetAddressSpace().get();
            gpu.InitChannel(*backend->channel, system.GetApplicationProcessProgramID());
            backend->channel->payload->maxwell_3d.SetNativeMacroState(graphics->GetState());
            backend->channel->payload->kepler_compute.SetNativeContext(graphics);
            std::scoped_lock lock{channel->mutex};
            channel->backend = std::move(backend);
            published = true;
        }
        auto& backend = *channel->backend;
        gpu.BindChannel(backend.channel->bind_id);
        activated_backend = true;
        gpu.InvalidateGPUCache();
        struct Host {
            NativeGpuChannel::Backend& backend;
            const Tegra::Control::NativeGpuGraphicsContext& graphics;
            Core::System& system;
            std::stop_token token;
            NativeGpuSubmission& submission;
        } host{backend, *graphics, system, token, *this};
        const Stream::Host bindings{
            &host,
            [](void* opaque, GPUVAddr address, size_t size) {
                return static_cast<Host*>(opaque)->graphics.GetAddressSpace()->IsReadableRange(
                    address, size);
            },
            [](void* opaque, GPUVAddr address, std::span<u32> words) {
                return static_cast<Host*>(opaque)->graphics.GetAddressSpace()->ReadBlockChecked(
                    address, {reinterpret_cast<u8*>(words.data()), words.size_bytes()});
            },
            [](void* opaque, u32 subchannel, u32 method, u32 value, bool last) {
                auto& self = *static_cast<Host*>(opaque);
                const auto class_id = self.backend.classes[subchannel];
                if ((class_id == 0xc797 || class_id == 0xc7c0) && method == 0x6c3) {
                    const bool compute = class_id == 0xc7c0;
                    std::optional<std::array<u32, 3>> report;
                    if (compute) {
                        report = self.backend.channel->payload->kepler_compute
                                     .GetNativeReportSemaphore();
                    } else {
                        const auto& engine = self.backend.channel->payload->maxwell_3d;
                        report = std::array{engine.GetRegisterValue(0x6c0),
                                            engine.GetRegisterValue(0x6c1),
                                            engine.GetRegisterValue(0x6c2)};
                    }
                    if (!report)
                        return false;
                    const auto upper = (*report)[0];
                    const auto lower = (*report)[1];
                    const GPUVAddr address = (GPUVAddr{upper} << 32) | lower;
                    const auto base = self.graphics.GetAddressSpace()->GetSyncpointShimAddress();
                    if (base && address >= *base &&
                        address - *base < Tegra::Host1x::SyncpointShim::Size) {
                        return !self.backend.PendingFrontend(self.graphics) &&
                               self.submission.ReleaseSyncpoint(self.system, upper, lower,
                                                                (*report)[2], value, self.token,
                                                                compute);
                    }
                }
                return self.backend.Call(self.system, self.graphics, subchannel, method, value,
                                         last, self.token);
            },
            [](void* opaque) {
                return static_cast<Host*>(opaque)
                    ->system.GPU()
                    .Renderer()
                    .ReadRasterizer()
                    ->SynchronizeNativeWork();
            },
            [](void* opaque) { return static_cast<Host*>(opaque)->token.stop_requested(); },
        };
        Stream stream;
        executed = stream.Execute(entries, bindings);
        if (executed.Succeeded() && backend.PendingFrontend(*graphics)) {
            executed.error = Stream::Error::MethodFailed;
        }
    } catch (const std::exception& error) {
        LOG_ERROR(HW_GPU, "Native GPU job execution failed: {}", error.what());
        executed.error = Stream::Error::MethodFailed;
    }
    // This also runs after allocation/renderer exceptions following a recorded dispatch.
    if (activated_backend) {
        try {
            auto& gpu = system.GPU();
            if (channel->backend->channel->payload->kepler_compute.IsNativeMode() &&
                !gpu.Renderer().ReadRasterizer()->SynchronizeNativeWork()) {
                executed.error = Stream::Error::BarrierFailed;
            }
            gpu.OnCommandListEnd();
        } catch (const std::exception& error) {
            LOG_ERROR(HW_GPU, "Native GPU prefix drain failed: {}", error.what());
            executed.error = Stream::Error::BarrierFailed;
        }
    }
    if (watchdog.joinable()) {
        watchdog.request_stop();
        watchdog.join();
    }
    const auto token = cancellation.get_token();
    if (token.stop_requested() && executed.Succeeded()) {
        executed.error = Stream::Error::Cancelled;
    }
    if (executed.Succeeded() && !fence_progress.IsComplete()) {
        // Submission metadata reserves stream increments; it never supplies missing work.
        executed.error = Stream::Error::MethodFailed;
    }
    const bool was_cancelled = token.stop_requested();
    if (!executed.Succeeded()) {
        {
            std::scoped_lock lock{channel->mutex};
            channel->faulted = true;
        }
        channel->cancellation.request_stop();
        // NVIDIA nvgpu UAPI notifier codes, not internal error-selector indices.
        const u32 information = timed_out.load()                             ? 8U
                                : executed.error == Stream::Error::ReadFault ? 31U
                                                                             : 13U;
        channel->GetErrorNotifier()->RecordError(information, system.GPU().GetTicks());
        LOG_ERROR(HW_GPU,
                  "Native GPU job failed: error={}, address={:#x}, subchannel={}, method={:#x}, "
                  "value={:#x}, "
                  "words={}, methods={}",
                  static_cast<u32>(executed.error), executed.address, executed.subchannel,
                  executed.method, executed.value, executed.words, executed.methods);
    } else {
        LOG_DEBUG(HW_GPU,
                  "Native GPU job retired: words={}, methods={}, parked_qmd={}, increments={}, "
                  "counter={}",
                  executed.words, executed.methods, graphics->HasPendingQmd(),
                  fence_progress.GetExecuted(),
                  system.Host1x().GetSyncpointManager().GetHostSyncpointValue(
                      static_cast<u32>(fence.id)));
    }
    Finish(executed.Succeeded() ? State::Complete
           : was_cancelled      ? State::Cancelled
                                : State::Failed,
           executed);
}

} // namespace Service::Nvidia::NvCore
