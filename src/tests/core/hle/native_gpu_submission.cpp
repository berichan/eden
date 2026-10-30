// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <atomic>
#include <barrier>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <thread>
#include <catch2/catch_test_macros.hpp>
#include "common/swap.h"
#include "core/core.h"
#include "core/device_memory.h"
#include "core/hle/service/nvdrv/core/native_gpu_backend.h"
#include "core/hle/service/nvdrv/core/native_gpu_submission.h"
#include "core/hle/service/nvdrv/devices/nvhost_native_gpu.h"
#include "video_core/control/channel_state_cache.inc"
#include "video_core/control/native_gpu_context.h"
#include "video_core/control/native_report_semaphore.h"
#include "video_core/gpu_thread.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/memory_manager.h"
#include "video_core/renderer_null/null_rasterizer.h"
#include "video_core/renderer_vulkan/vk_state_tracker.h"
#ifdef HAS_OPENGL
#include "video_core/renderer_opengl/gl_state_tracker.h"
#endif

namespace Service::Nvidia::NvCore {
struct NativeGpuBackendTest {
    static void Run(bool compute_upload) {
        Core::System system;
        Core::DeviceMemory memory;
        Tegra::MaxwellDeviceMemoryManager backing{memory};
        auto space =
            std::make_shared<Tegra::MemoryManager>(system, backing, 32, 1ULL << 32, 16, 12);
        Tegra::GPU gpu{system, false, false};
        Null::RasterizerNull rasterizer{gpu};
        backing.BindInterface(&rasterizer);
        auto graphics = std::make_shared<Tegra::Control::AmpereGraphicsState>();
        auto context = std::make_shared<Tegra::Control::NativeGpuGraphicsContext>(graphics, space);
        NativeGpuChannel::Backend backend;
        backend.channel = std::make_shared<Tegra::Control::ChannelState>(0);
        backend.channel->memory_manager = space.get();
        backend.channel->Init(system, 0);
        backend.channel->BindRasterizer(&rasterizer);
        auto& payload = *backend.channel->payload;
        payload.maxwell_3d.SetNativeMacroState(graphics);
        payload.kepler_compute.SetNativeContext(context);
        const auto call = [&](u32 subchannel, u32 method, u32 value) {
            return backend.CallWithGpu(system, gpu, rasterizer, *context, subchannel, method, value,
                                       true, {});
        };
        REQUIRE(call(0, 0, 0xc797));
        REQUIRE(call(1, 0, 0xc7c0));
        if (compute_upload) {
            REQUIRE(space->MapSparse(0x8000000, 0x1000, false) == 0x8000000);
            REQUIRE(call(1, 0x60, 8));
            REQUIRE(call(1, 0x61, 1));
            REQUIRE(call(1, 0x62, 0));
            REQUIRE(call(1, 0x63, 0x8000000));
            REQUIRE(call(1, 0x64, 0));
            REQUIRE(call(1, 0x6c, 0x1011));
            REQUIRE(call(1, 0x6d, 0x12345678));
            REQUIRE(payload.kepler_compute.HasPendingNativeInlineUpload());
        } else {
            REQUIRE(call(0, 0x52, 0));
            REQUIRE(backend.atomic_sequence.IsActive());
        }
        REQUIRE(backend.PendingFrontend(*context));
        REQUIRE_FALSE(call(0, 0x378, 2));
        REQUIRE_FALSE(call(0, 0x586, 4));
        REQUIRE_FALSE(payload.maxwell_3d.HasPendingNativeDraw());
        REQUIRE_FALSE(payload.maxwell_3d.HasNativeMacroFailure());
        REQUIRE(payload.maxwell_3d.GetRegisterValue(0x586) == 0);
        if (!compute_upload) {
            REQUIRE(call(0, 0x53, 0));
            REQUIRE_FALSE(backend.PendingFrontend(*context));
            REQUIRE(call(0, 0x651, 2));
            REQUIRE(call(0, 0x8f, 3));
            REQUIRE(call(0, 0x586, 4));
            REQUIRE(backend.PendingFrontend(*context));
            REQUIRE(call(0, 0x5f8, 3));
            REQUIRE(backend.PendingFrontend(*context));
            REQUIRE_FALSE(call(0, 0x6c3, 0xf006));
            REQUIRE(call(0, 0x585, 0));
            REQUIRE_FALSE(backend.PendingFrontend(*context));
        }
    }
};
} // namespace Service::Nvidia::NvCore

TEST_CASE("Native backend rejects graphics BEGIN inside an atomic scope", "[core][ipc]") {
    Service::Nvidia::NvCore::NativeGpuBackendTest::Run(false);
}

TEST_CASE("Native backend rejects graphics BEGIN across incomplete compute uploads",
          "[core][ipc]") {
    Service::Nvidia::NvCore::NativeGpuBackendTest::Run(true);
}

namespace {
using namespace Service::Nvidia;
struct LifetimeInfo : VideoCommon::ChannelInfo {
    inline static u32 live{};
    explicit LifetimeInfo(Tegra::Control::ChannelState& channel) : ChannelInfo{channel} {
        ++live;
    }
    ~LifetimeInfo() {
        --live;
    }
};
struct FailingRegistration : VideoCommon::ChannelSetupCaches<LifetimeInfo> {
    bool fail{true};
    void OnGPUASRegister(size_t) override {
        if (fail) {
            throw std::runtime_error("registration failed");
        }
    }
};
struct Fixture {
    Core::System system;
    Core::DeviceMemory memory;
    Tegra::MaxwellDeviceMemoryManager backing{memory};
    std::shared_ptr<Tegra::MemoryManager> space =
        std::make_shared<Tegra::MemoryManager>(system, backing, 32, 1ULL << 32, 16, 12);
    Tegra::Host1x::SyncpointManager syncpoints;
    NvCore::NativeGpuChannels channels{syncpoints};
    const NvCore::SessionId owner{7};
    std::shared_ptr<NvCore::NativeGpuSchedulingGroup> group =
        std::make_shared<NvCore::NativeGpuSchedulingGroup>(owner);
    std::shared_ptr<NvCore::NativeGpuChannel> channel;
    explicit Fixture(u32 initial_fence = 0) {
        REQUIRE(channels.Allocate(4, owner, 0) == NvResult::Success);
        channel = channels.Get(4, owner);
        REQUIRE(channel->BindAddressSpace(space) == NvResult::Success);
        REQUIRE(channel->BindGroup(group, 0) == NvResult::Success);
        REQUIRE(channels.AllocateObjectContext(4, owner, 0xc797, 0) == NvResult::Success);
        if (initial_fence) {
            REQUIRE(syncpoints.PublishNativeCompletion(17, initial_fence, initial_fence));
        }
        NvFence fence;
        REQUIRE(channels.AllocateCommandRing(&fence, 4, owner, 8, 3, 0) == NvResult::Success);
    }
    NvResult Reserve(std::shared_ptr<NvCore::NativeGpuSubmission>& job, size_t count = 1,
                     u32 flags = 0x104, u32 increment = 0) {
        const std::vector<u64> entries(count, 0);
        NvFence fence{-1, 0xcccccccc};
        const auto result =
            channels.ReserveSubmission(&job, &fence, 4, owner, entries, flags, increment);
        if (result == NvResult::Success) {
            REQUIRE(fence.id == 17);
            REQUIRE(fence.value == 0);
            REQUIRE(syncpoints.GetHostSyncpointValue(17) == 0);
            REQUIRE(syncpoints.GetGuestSyncpointValue(17) == 0);
        }
        return result;
    }
};
} // namespace

TEST_CASE("Native jobs reserve ring capacity without signalling completion", "[core][ipc]") {
    Fixture fixture;
    using Job = NvCore::NativeGpuSubmission;
    std::shared_ptr<Job> first, second, third, blocked;
    REQUIRE(fixture.Reserve(first, 3) == NvResult::Success);
    REQUIRE(fixture.Reserve(second, 3) == NvResult::Success);
    REQUIRE(fixture.Reserve(third, 1) == NvResult::Success);
    REQUIRE(fixture.Reserve(blocked) == NvResult::Busy);
    REQUIRE(fixture.channel->UnbindGroup(fixture.group) == NvResult::Busy);
    REQUIRE_FALSE(second->MarkQueued());
    REQUIRE(first->MarkQueued());
    REQUIRE_FALSE(first->MarkQueued());
    REQUIRE(second->MarkQueued());
    second->Cancel();
    REQUIRE(second->GetState() == Job::State::Cancelled);
    REQUIRE(fixture.Reserve(blocked) == NvResult::Busy);
    first->Cancel();
    REQUIRE(fixture.Reserve(blocked, 6) == NvResult::Success);
    REQUIRE(blocked->GetEntries().size() == 6);
    REQUIRE(fixture.Reserve(blocked) == NvResult::BadParameter);
    third.reset();
    blocked.reset();
    std::shared_ptr<Job> reused;
    REQUIRE(fixture.Reserve(reused, 7) == NvResult::Success);
    reused->Cancel();
    REQUIRE(fixture.channel->UnbindGroup(fixture.group) == NvResult::Success);
}

TEST_CASE("Queued native jobs retain resources and cancel without GPU initialization",
          "[core][ipc]") {
    Fixture fixture;
    using Job = NvCore::NativeGpuSubmission;
    std::shared_ptr<Job> job;
    const std::array<u64, 1> entries{};
    NvFence fence{};
    REQUIRE(fixture.channels.ReserveSubmission(&job, &fence, 4, fixture.owner, entries, 0x104, 1) ==
            NvResult::Success);
    REQUIRE(fence.value == 1);
    REQUIRE(job->MarkQueued());
    const std::weak_ptr<Tegra::MemoryManager> retained_space = fixture.space;
    fixture.channels.Close(4, fixture.owner);
    fixture.space.reset();
    fixture.group.reset();
    fixture.channel.reset();
    REQUIRE_FALSE(retained_space.expired());
    REQUIRE(fixture.channels.Allocate(5, fixture.owner, 0) == NvResult::Success);
    REQUIRE(fixture.channels.Get(5, fixture.owner)->GetChannelIndex() == 1);
    job->Execute(fixture.system, {});
    REQUIRE(job->GetState() == Job::State::Cancelled);
    REQUIRE(job->GetResult().error == Tegra::Control::NativeCommandStream::Error::Cancelled);
    REQUIRE(fixture.syncpoints.GetHostSyncpointValue(17) == 0);
    REQUIRE(fixture.syncpoints.GetGuestSyncpointValue(17) == 0);
    job.reset();
    REQUIRE(retained_space.expired());
    REQUIRE(fixture.channels.Allocate(6, fixture.owner, 0) == NvResult::Success);
    REQUIRE(fixture.channels.Get(6, fixture.owner)->GetChannelIndex() == 0);
}

TEST_CASE("Native queue rejection and ioctl validation preserve outputs and reservations",
          "[core][ipc]") {
    Fixture fixture;
    std::shared_ptr<NvCore::NativeGpuSubmission> job;
    for (u32 flags : {1U, 2U, 8U, 0xffffffffU}) {
        REQUIRE(fixture.Reserve(job, 1, flags) == NvResult::NotSupported);
        REQUIRE_FALSE(job);
    }
    REQUIRE(fixture.Reserve(job, 1, 0x104, 65) == NvResult::BadParameter);
    REQUIRE(fixture.Reserve(job, 1, 4, 1) == NvResult::NotSupported);
    REQUIRE(fixture.Reserve(job, 0) == NvResult::BadParameter);
    REQUIRE(fixture.Reserve(job) == NvResult::Success);
    auto queue = std::make_unique<VideoCommon::GPUThread::ThreadManager>(fixture.system);
    REQUIRE_FALSE(queue->SubmitNative(job));
    REQUIRE(job->GetState() == NvCore::NativeGpuSubmission::State::Reserved);
    job.reset();
    Devices::nvhost_native_gpu device{fixture.system, fixture.channels};
    device.OnOpen(fixture.owner, 4);
    std::array<u8, 24> input{}, output;
    output.fill(0xcc);
    const std::array<u64_le, 1> inline_entries{};
    const auto inl =
        std::span{reinterpret_cast<const u8*>(inline_entries.data()), sizeof(inline_entries)};
    const auto set = [&](u32 word, u32 value) {
        const u32_le wire{value};
        std::memcpy(input.data() + word * 4, &wire, 4);
    };
    const Ioctl submit{.raw = 0xc018481b};
    REQUIRE(device.Ioctl2(3, submit, input, inl, output) == NvResult::InvalidState);
    REQUIRE(device.Ioctl2(4, Ioctl{.raw = 0x4018481b}, input, inl, output) ==
            NvResult::BadParameter);
    REQUIRE(device.Ioctl2(4, Ioctl{.raw = 0xc014481b}, input, inl, output) ==
            NvResult::InvalidSize);
    set(2, 2);
    REQUIRE(device.Ioctl2(4, submit, input, inl, output) == NvResult::InvalidSize);
    set(2, 1);
    set(3, 2);
    REQUIRE(device.Ioctl2(4, submit, input, inl, output) == NvResult::NotSupported);
    set(3, 0x104);
    REQUIRE(device.Ioctl2(4, submit, input, inl, output) == NvResult::NotInitialized);
    REQUIRE(output.front() == 0xcc);
    REQUIRE(output.back() == 0xcc);
    REQUIRE(fixture.Reserve(job, 7) == NvResult::Success);
}

TEST_CASE("Native counted fences reserve targets and roll back failed queue publication",
          "[core][ipc]") {
    Fixture fixture;
    using Job = NvCore::NativeGpuSubmission;
    const std::array<u64, 1> entries{};
    std::shared_ptr<Job> first, second, third;
    NvFence fence{-1, 0xcccccccc};
    const auto reserve = [&](auto& job, u32 flags, u32 increments) {
        return fixture.channels.ReserveSubmission(&job, &fence, 4, fixture.owner, entries, flags,
                                                  increments);
    };
    REQUIRE(reserve(first, 0x104, 65) == NvResult::BadParameter);
    REQUIRE(fence.id == -1);
    REQUIRE(fence.value == 0xcccccccc);
    REQUIRE(reserve(first, 0x104, 1) == NvResult::Success);
    REQUIRE(fence.value == 1);
    REQUIRE(reserve(second, 0x114, 2) == NvResult::Success);
    REQUIRE(fence.value == 3);
    REQUIRE(fixture.syncpoints.GetHostSyncpointValue(17) == 0);
    REQUIRE(fixture.syncpoints.GetGuestSyncpointValue(17) == 0);
    REQUIRE(first->MarkQueued());
    REQUIRE(second->MarkQueued());
    second->CancelUnpublished();
    REQUIRE(second->GetState() == Job::State::Cancelled);
    REQUIRE(reserve(third, 0x104, 4) == NvResult::Success);
    REQUIRE(fence.value == 5);
    third.reset();
    REQUIRE(reserve(third, 0x104, 1) == NvResult::Success);
    REQUIRE(fence.value == 2);
    third.reset();
    REQUIRE(reserve(third, 0x104, 64) == NvResult::Success);
    REQUIRE(fence.value == 65);
    third.reset();
    first->Cancel();
    REQUIRE(reserve(third, 0x104, 1) == NvResult::InvalidState);
    REQUIRE(fixture.syncpoints.GetHostSyncpointValue(17) == 0);
    REQUIRE(fixture.syncpoints.GetGuestSyncpointValue(17) == 0);
}

TEST_CASE("Native counted reservations reject overflow and reuse unpublished tail targets",
          "[core][ipc]") {
    Fixture fixture{std::numeric_limits<u32>::max() - 1};
    const std::array<u64, 1> entries{};
    std::shared_ptr<NvCore::NativeGpuSubmission> job;
    NvFence fence{};
    REQUIRE(fixture.channels.ReserveSubmission(&job, &fence, 4, fixture.owner, entries, 0x104, 2) ==
            NvResult::OverFlow);
    REQUIRE(fixture.channels.ReserveSubmission(&job, &fence, 4, fixture.owner, entries, 0x104, 1) ==
            NvResult::Success);
    REQUIRE(fence.value == std::numeric_limits<u32>::max());
    job.reset();
    REQUIRE(fixture.channels.ReserveSubmission(&job, &fence, 4, fixture.owner, entries, 0x114, 1) ==
            NvResult::Success);
    REQUIRE(fence.value == std::numeric_limits<u32>::max());
    REQUIRE(job->MarkQueued());
    job->CancelUnpublished();
    job.reset();
    REQUIRE(fixture.channels.ReserveSubmission(&job, &fence, 4, fixture.owner, entries, 4, 0) ==
            NvResult::Success);
    REQUIRE(fence.value == std::numeric_limits<u32>::max() - 1);
}

TEST_CASE("Cancelling an earlier native counted reservation cannot bridge a fence gap",
          "[core][ipc]") {
    Fixture fixture;
    const std::array<u64, 1> entries{};
    std::shared_ptr<NvCore::NativeGpuSubmission> first, second, third;
    NvFence fence{};
    REQUIRE(fixture.channels.ReserveSubmission(&first, &fence, 4, fixture.owner, entries, 0x104,
                                               1) == NvResult::Success);
    REQUIRE(fixture.channels.ReserveSubmission(&second, &fence, 4, fixture.owner, entries, 0x104,
                                               0) == NvResult::Success);
    REQUIRE(fence.value == 1);
    first->Cancel();
    REQUIRE_FALSE(second->MarkQueued());
    REQUIRE(fixture.channels.ReserveSubmission(&third, &fence, 4, fixture.owner, entries, 0x104,
                                               1) == NvResult::InvalidState);
    REQUIRE(fixture.syncpoints.GetHostSyncpointValue(17) == 0);
    REQUIRE(fixture.syncpoints.GetGuestSyncpointValue(17) == 0);
}

TEST_CASE("Native fence metadata is consumed only by executed stream increments", "[core][ipc]") {
    Tegra::Host1x::SyncpointManager syncpoints;
    NvCore::NativeGpuFenceProgress progress{17, 2, 2};
    REQUIRE_FALSE(progress.IsComplete());
    REQUIRE(progress.HasRemaining());
    REQUIRE(syncpoints.GetHostSyncpointValue(17) == 0);
    REQUIRE(progress.Signal(syncpoints));
    REQUIRE(progress.GetExecuted() == 1);
    REQUIRE_FALSE(progress.IsComplete());
    REQUIRE(syncpoints.GetHostSyncpointValue(17) == 1);
    REQUIRE(syncpoints.GetGuestSyncpointValue(17) == 1);
    // Inspecting or failing a later stream prefix cannot fill the missing increment.
    REQUIRE_FALSE(progress.IsComplete());
    REQUIRE(syncpoints.GetHostSyncpointValue(17) == 1);
    REQUIRE(progress.Signal(syncpoints));
    REQUIRE(progress.IsComplete());
    REQUIRE_FALSE(progress.HasRemaining());
    REQUIRE_FALSE(progress.Signal(syncpoints));
    REQUIRE(progress.GetExecuted() == 2);
    REQUIRE(syncpoints.GetHostSyncpointValue(17) == 2);
    REQUIRE(syncpoints.GetGuestSyncpointValue(17) == 2);
    REQUIRE(syncpoints.GetHostSyncpointValue(18) == 0);
    NvCore::NativeGpuFenceProgress empty{17, 2, 0};
    REQUIRE(empty.IsComplete());
    REQUIRE_FALSE(empty.Signal(syncpoints));
    NvCore::NativeGpuFenceProgress invalid{17, 1, 2};
    REQUIRE_FALSE(invalid.HasRemaining());
    REQUIRE_FALSE(invalid.IsComplete());
    REQUIRE_FALSE(invalid.Signal(syncpoints));
    NvCore::NativeGpuFenceProgress wrong_order{17, 4, 1};
    REQUIRE_FALSE(wrong_order.Signal(syncpoints));
    REQUIRE(wrong_order.GetExecuted() == 0);
    REQUIRE(syncpoints.GetHostSyncpointValue(17) == 2);
}

TEST_CASE("Native shim report guards reject foreign writes and retain completed prefix increments",
          "[core][ipc]") {
    using Report = Tegra::Control::NativeReportSemaphore;
    struct Context {
        Tegra::Host1x::SyncpointManager syncpoints;
        NvCore::NativeGpuFenceProgress progress{17, 2, 2};
        GPUVAddr owned{0xc000110000};
        bool stopped{}, cancel_in_wait{};
        u32 waits{}, writes{};
        Report::Host Bind() {
            return {this,
                    [](void* opaque, GPUVAddr address, size_t size) {
                        auto& self = *static_cast<Context*>(opaque);
                        return self.progress.CanWriteCounter(self.owned, address, size);
                    },
                    [](void* opaque) {
                        auto& self = *static_cast<Context*>(opaque);
                        ++self.waits;
                        self.stopped = self.cancel_in_wait;
                        return true;
                    },
                    [](void* opaque, GPUVAddr, u32) {
                        auto& self = *static_cast<Context*>(opaque);
                        ++self.writes;
                        return self.progress.Signal(self.syncpoints);
                    },
                    [](void* opaque) { return static_cast<Context*>(opaque)->stopped; }};
        }
    };
    Context guarded;
    for (u32 lower : {0x120000U, 0x110004U}) {
        REQUIRE(Report::Release(0xc0, lower, 1, 0x1000f010, guarded.Bind()) ==
                Report::Error::MemoryFault);
    }
    REQUIRE_FALSE(guarded.progress.CanWriteCounter(guarded.owned, guarded.owned, 8));
    REQUIRE(guarded.waits == 0);
    REQUIRE(guarded.writes == 0);
    Context cancelled;
    cancelled.cancel_in_wait = true;
    REQUIRE(Report::Release(0xc0, 0x110000, 1, 0x1000f010, cancelled.Bind()) ==
            Report::Error::Cancelled);
    REQUIRE(cancelled.writes == 0);
    REQUIRE(cancelled.syncpoints.GetHostSyncpointValue(17) == 0);
    Context prefix;
    using Stream = Tegra::Control::NativeCommandStream;
    Stream stream;
    Stream::Host host{};
    host.context = &prefix;
    host.method = [](void* opaque, u32, u32 method, u32 value, bool) {
        auto& self = *static_cast<Context*>(opaque);
        return method == 0x6c3 && Report::Release(0xc0, 0x110000, 0xffffffff, value, self.Bind()) ==
                                      Report::Error::None;
    };
    const std::array<u32, 4> words{0x200106c3, 0x1000f010, 0x20010100, 0};
    REQUIRE(stream.Process(words, 0x1000, host).error == Stream::Error::MethodFailed);
    REQUIRE(prefix.progress.GetExecuted() == 1);
    REQUIRE_FALSE(prefix.progress.IsComplete());
    REQUIRE(prefix.syncpoints.GetHostSyncpointValue(17) == 1);
    REQUIRE(prefix.syncpoints.GetGuestSyncpointValue(17) == 1);
    REQUIRE(Report::Release(0xc0, 0x110000, 0, 0x1000f010, prefix.Bind()) == Report::Error::None);
    REQUIRE(prefix.progress.IsComplete());
    REQUIRE(prefix.syncpoints.GetHostSyncpointValue(17) == 2);
    REQUIRE(Report::Release(0xc0, 0x110000, 1, 0x1000f010, prefix.Bind()) ==
            Report::Error::MemoryFault);
    REQUIRE(prefix.syncpoints.GetHostSyncpointValue(17) == 2);
}

TEST_CASE("Native completion publishes both counters before all ready notifications",
          "[core][ipc]") {
    Tegra::Host1x::SyncpointManager syncpoints;
    bool guest_seen{}, host_seen{}, after_throw{}, future_seen{};
    const auto check_counters = [&] {
        REQUIRE(syncpoints.GetGuestSyncpointValue(17) == 3);
        REQUIRE(syncpoints.GetHostSyncpointValue(17) == 3);
    };
    syncpoints.RegisterGuestAction(17, 1, [&] {
        check_counters();
        guest_seen = true;
    });
    syncpoints.RegisterGuestAction(17, 2, [] { throw std::runtime_error("notification failed"); });
    syncpoints.RegisterGuestAction(17, 3, [&] { after_throw = true; });
    syncpoints.RegisterHostAction(17, 3, [&] {
        check_counters();
        host_seen = true;
    });
    syncpoints.RegisterHostAction(17, 4, [&] { future_seen = true; });
    REQUIRE_FALSE(syncpoints.PublishNativeCompletion(17, 3, 2));
    REQUIRE(syncpoints.GetGuestSyncpointValue(17) == 0);
    REQUIRE(syncpoints.GetHostSyncpointValue(17) == 0);
    REQUIRE(syncpoints.PublishNativeCompletion(17, 3, 3));
    REQUIRE(guest_seen);
    REQUIRE(host_seen);
    REQUIRE(after_throw);
    REQUIRE_FALSE(future_seen);
    REQUIRE_FALSE(syncpoints.PublishNativeCompletion(17, 5, 1));
    REQUIRE_FALSE(syncpoints.PublishNativeCompletion(17, 0, 1));
    REQUIRE_FALSE(syncpoints.PublishNativeCompletion(17, 3, 0));
    REQUIRE_FALSE(syncpoints.PublishNativeCompletion(448, 1, 1));
    syncpoints.IncrementGuest(17);
    syncpoints.IncrementHost(17);
    REQUIRE(future_seen);
    REQUIRE(syncpoints.PublishNativeCompletion(17, 5, 1));
}

TEST_CASE("Concurrent native fence registration observes the published host counter",
          "[core][ipc]") {
    Tegra::Host1x::SyncpointManager syncpoints;
    constexpr u32 Count = 10000;
    std::barrier start{2};
    std::atomic<bool> ordered{true};
    std::atomic<u32> notified{};
    std::jthread registration([&] {
        start.arrive_and_wait();
        for (u32 target = 1; target <= Count; ++target) {
            syncpoints.RegisterGuestAction(17, target, [&, target] {
                if (syncpoints.GetHostSyncpointValue(17) < target) {
                    ordered.store(false);
                }
                notified.fetch_add(1);
            });
        }
    });
    start.arrive_and_wait();
    bool published = true;
    for (u32 target = 1; target <= Count; ++target) {
        published &= syncpoints.PublishNativeCompletion(17, target, 1);
    }
    registration.join();
    REQUIRE(published);
    REQUIRE(ordered.load());
    REQUIRE(notified.load() == Count);
}

TEST_CASE("Retired channel caches release state and preserve distinct address-space slots",
          "[core][ipc]") {
    Fixture fixture;
    auto second_space = std::make_shared<Tegra::MemoryManager>(fixture.system, fixture.backing, 32,
                                                               1ULL << 32, 16, 12);
    auto first = std::make_unique<Tegra::Control::ChannelState>(0);
    auto shared = std::make_unique<Tegra::Control::ChannelState>(1);
    auto other = std::make_unique<Tegra::Control::ChannelState>(2);
    first->memory_manager = shared->memory_manager = fixture.space.get();
    other->memory_manager = second_space.get();
    for (auto* channel : {first.get(), shared.get(), other.get()})
        channel->Init(fixture.system, 0);
    FailingRegistration failed;
    REQUIRE_THROWS_AS(failed.CreateChannel(*first), std::runtime_error);
    REQUIRE(LifetimeInfo::live == 0);
    REQUIRE(failed.GetFromID(fixture.space->GetID()) == nullptr);
    failed.fail = false;
    failed.CreateChannel(*first);
    REQUIRE(LifetimeInfo::live == 1);
    failed.EraseChannel(0);
    REQUIRE(LifetimeInfo::live == 0);
    VideoCommon::ChannelSetupCaches<LifetimeInfo> cache;
    cache.CreateChannel(*first);
    cache.CreateChannel(*shared);
    REQUIRE(LifetimeInfo::live == 2);
    const size_t first_id = fixture.space->GetID();
    cache.EraseChannel(0);
    REQUIRE(LifetimeInfo::live == 1);
    REQUIRE(cache.GetFromID(first_id) == fixture.space.get());
    cache.EraseChannel(1);
    REQUIRE(LifetimeInfo::live == 0);
    REQUIRE(cache.GetFromID(first_id) == nullptr);
    REQUIRE_FALSE(cache.getStorageID(first_id));
    cache.CreateChannel(*other);
    cache.CreateChannel(*first);
    REQUIRE(LifetimeInfo::live == 2);
    REQUIRE(cache.GetFromID(first_id) == fixture.space.get());
    REQUIRE(cache.getStorageID(first_id) != cache.getStorageID(second_space->GetID()));
    cache.EraseChannel(0);
    cache.EraseChannel(2);
    cache.EraseChannel(2);
    REQUIRE(LifetimeInfo::live == 0);
    const auto detach = [&](auto& tracker) {
        auto& flags = first->payload->maxwell_3d.dirty.flags;
        tracker.ChangeChannel(*first);
        tracker.ReleaseChannel(99);
        flags.reset();
        tracker.InvalidateState();
        REQUIRE(flags.any());
        tracker.ReleaseChannel(0);
        flags.reset();
        tracker.InvalidateState();
        REQUIRE(flags.none());
    };
    Vulkan::StateTracker vulkan;
    detach(vulkan);
#ifdef HAS_OPENGL
    OpenGL::StateTracker opengl;
    detach(opengl);
#endif
}
