// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstring>
#include <stdexcept>
#include <catch2/catch_test_macros.hpp>
#include "common/swap.h"
#include "core/core.h"
#include "core/device_memory.h"
#include "core/hle/service/nvdrv/core/native_gpu_submission.h"
#include "core/hle/service/nvdrv/devices/nvhost_native_gpu.h"
#include "video_core/control/channel_state_cache.inc"
#include "video_core/gpu_thread.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/memory_manager.h"
#include "video_core/renderer_vulkan/vk_state_tracker.h"
#ifdef HAS_OPENGL
#include "video_core/renderer_opengl/gl_state_tracker.h"
#endif

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
    Fixture() {
        REQUIRE(channels.Allocate(4, owner, 0) == NvResult::Success);
        channel = channels.Get(4, owner);
        REQUIRE(channel->BindAddressSpace(space) == NvResult::Success);
        REQUIRE(channel->BindGroup(group, 0) == NvResult::Success);
        REQUIRE(channels.AllocateObjectContext(4, owner, 0xc797, 0) == NvResult::Success);
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
    REQUIRE(fixture.Reserve(job) == NvResult::Success);
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
    REQUIRE(fixture.Reserve(job, 1, 0x104, 1) == NvResult::NotSupported);
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
