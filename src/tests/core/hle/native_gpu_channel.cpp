// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstring>
#include <catch2/catch_test_macros.hpp>
#include "common/swap.h"
#include "core/core.h"
#include "core/device_memory.h"
#include "core/hle/service/nvdrv/core/native_gpu_channel.h"
#include "core/hle/service/nvdrv/devices/native_gpu_characteristics.h"
#include "core/hle/service/nvdrv/devices/nvhost_native_gpu.h"
#include "core/hle/service/nvdrv/devices/nvhost_tsg_gpu.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/memory_manager.h"

TEST_CASE("Native channel allocation creates owned runlist resources after a valid request",
          "[core][ipc]") {
    using namespace Service::Nvidia;
    Core::System system;
    NvCore::NativeGpuChannels channels;
    Devices::nvhost_native_gpu device{system, channels};
    const NvCore::SessionId owner{7};
    const NvCore::SessionId foreign{8};
    const Ioctl allocate{.raw = 0x40044820};
    std::array<u8, 8> input{};
    std::array<u8, 8> output{};
    output.fill(0xcc);
    REQUIRE(device.Ioctl1(4, allocate, input, output) == NvResult::InvalidState);
    device.OnOpen(owner, 4);
    REQUIRE(channels.Get(4, owner) == nullptr);
    REQUIRE(device.Ioctl1(5, allocate, input, output) == NvResult::InvalidState);
    for (u32 raw : {0x00044820U, 0x80044820U, 0xc0044820U}) {
        REQUIRE(device.Ioctl1(4, Ioctl{.raw = raw}, input, output) == NvResult::BadParameter);
    }
    for (u32 raw : {0x40004820U, 0x40084820U}) {
        REQUIRE(device.Ioctl1(4, Ioctl{.raw = raw}, input, output) == NvResult::InvalidSize);
    }
    REQUIRE(device.Ioctl1(4, allocate, std::span{input}.first(3), output) == NvResult::InvalidSize);
    REQUIRE(device.Ioctl1(4, Ioctl{.raw = 0xc020481a}, input, output) == NvResult::NotSupported);
    REQUIRE(channels.Get(4, owner) == nullptr);
    REQUIRE(device.Ioctl1(4, allocate, input, output) == NvResult::Success);
    auto channel = channels.Get(4, owner);
    REQUIRE(channel != nullptr);
    REQUIRE(channel->GetRunlist() == 0);
    REQUIRE(channel->IsOpen());
    REQUIRE(channel->GetAddressSpace() == nullptr);
    REQUIRE(channel->GetGroup() == nullptr);
    REQUIRE_FALSE(channel->GetSubcontext());
    REQUIRE(channels.Get(4, foreign) == nullptr);
    REQUIRE(channels.Get(3, owner) == nullptr);
    REQUIRE(device.Ioctl1(4, allocate, input, output) == NvResult::InvalidState);
    REQUIRE(output.front() == 0xcc);
    REQUIRE(output.back() == 0xcc);
    REQUIRE(device.Ioctl2(4, allocate, input, input, output) == NvResult::NotSupported);
    REQUIRE(device.Ioctl3(4, allocate, input, output, output) == NvResult::NotSupported);
    channels.Close(4, foreign);
    REQUIRE(channel->IsOpen());
    device.OnClose(5);
    REQUIRE(channels.Get(4, owner) == channel);
    device.OnClose(4);
    REQUIRE(channels.Get(4, owner) == nullptr);
    REQUIRE_FALSE(channel->IsOpen());
    REQUIRE(device.Ioctl1(4, allocate, input, output) == NvResult::InvalidState);
    REQUIRE(device.Ioctl2(4, allocate, input, input, output) == NvResult::InvalidState);
    REQUIRE(device.Ioctl3(4, allocate, input, output, output) == NvResult::InvalidState);

    device.OnOpen(owner, 6);
    input[0] = 1;
    REQUIRE(device.Ioctl1(6, allocate, input, output) == NvResult::Success);
    REQUIRE(channels.Get(6, owner)->GetRunlist() == 1);
    REQUIRE(channels.Allocate(-1, owner, 0) == NvResult::BadParameter);
    REQUIRE(channels.Allocate(7, owner, 0xffffffff) == NvResult::Success);
    REQUIRE(channels.Get(7, owner)->GetRunlist() == 0);
    REQUIRE(channels.Allocate(7, foreign, 1) == NvResult::InvalidState);
}

TEST_CASE("Native channel groups enforce runlist and per-subcontext address-space identity",
          "[core][ipc]") {
    using namespace Service::Nvidia;
    Core::System system;
    Core::DeviceMemory memory;
    Tegra::MaxwellDeviceMemoryManager backing{memory};
    auto first_space =
        std::make_shared<Tegra::MemoryManager>(system, backing, 32, 1ULL << 32, 16, 12);
    auto second_space =
        std::make_shared<Tegra::MemoryManager>(system, backing, 32, 1ULL << 32, 16, 12);
    const NvCore::SessionId owner{7};
    NvCore::NativeGpuChannels channels;
    auto group = std::make_shared<NvCore::NativeGpuSchedulingGroup>(owner);
    auto foreign_group = std::make_shared<NvCore::NativeGpuSchedulingGroup>(NvCore::SessionId{8});
    REQUIRE(Devices::NativeGpuCharacteristics{}.unknown_b4 ==
            NvCore::NativeGpuSchedulingGroup::MaxSubcontexts);
    for (DeviceFD fd = 1; fd <= 4; ++fd) {
        REQUIRE(channels.Allocate(fd, owner, fd == 4 ? 1 : 0) == NvResult::Success);
    }
    auto first = channels.Get(1, owner);
    auto second = channels.Get(2, owner);
    auto third = channels.Get(3, owner);
    auto copy = channels.Get(4, owner);
    REQUIRE(first->BindGroup(group, 0) == NvResult::InvalidState);
    REQUIRE(first->BindAddressSpace(nullptr) == NvResult::BadParameter);
    REQUIRE(first->BindAddressSpace(first_space) == NvResult::Success);
    REQUIRE(first->GetAddressSpace() == first_space);
    REQUIRE(first->BindAddressSpace(second_space) == NvResult::InvalidState);
    REQUIRE(first->BindGroup(foreign_group, 0) == NvResult::AccessDenied);
    REQUIRE(first->BindGroup(group, 64) == NvResult::BadParameter);
    REQUIRE(group->GetChannelCount() == 0);
    REQUIRE_FALSE(group->GetRunlist());
    REQUIRE(first->BindGroup(group, 1) == NvResult::Success);
    REQUIRE(first->GetSubcontext() == 1U);
    REQUIRE(group->GetRunlist() == 0U);
    REQUIRE(first->BindGroup(group, 1) == NvResult::InvalidState);
    REQUIRE(second->BindAddressSpace(first_space) == NvResult::Success);
    REQUIRE(second->BindGroup(group, 1) == NvResult::Success);
    REQUIRE(third->BindAddressSpace(second_space) == NvResult::Success);
    REQUIRE(third->BindGroup(group, 1) == NvResult::BadParameter);
    REQUIRE(third->GetGroup() == nullptr);
    REQUIRE(third->BindGroup(group, 63) == NvResult::Success);
    REQUIRE(copy->BindAddressSpace(first_space) == NvResult::Success);
    REQUIRE(copy->BindGroup(group, 2) == NvResult::BadParameter);
    REQUIRE(group->GetChannelCount() == 3);
    REQUIRE(group->SetInterleave(2) == NvResult::Success);
    REQUIRE(group->SetL2Promotion(4) == NvResult::Success);
    REQUIRE(first->GetGroup()->GetParameters().interleave_level == 2);
    REQUIRE(second->GetGroup()->GetParameters().l2_promotion == 4);
    REQUIRE(first->UnbindGroup(foreign_group) == NvResult::BadParameter);
    REQUIRE(first->UnbindGroup(group) == NvResult::Success);
    REQUIRE_FALSE(first->GetSubcontext());
    REQUIRE(group->GetChannelCount() == 2);
    channels.Close(2, owner);
    REQUIRE(group->GetChannelCount() == 2);
    REQUIRE(second->UnbindGroup(group) == NvResult::InvalidState);
    second.reset();
    REQUIRE(group->GetChannelCount() == 1);
    REQUIRE(third->UnbindGroup(group) == NvResult::Success);
    REQUIRE(group->GetChannelCount() == 0);
    REQUIRE(group->GetRunlist() == 0U);
    REQUIRE(copy->BindGroup(group, 2) == NvResult::BadParameter);
}

TEST_CASE("Native TSG bind ioctls retain group and address space beyond descriptor closure",
          "[core][ipc]") {
    using namespace Service::Nvidia;
    Core::System system;
    Core::DeviceMemory memory;
    Tegra::MaxwellDeviceMemoryManager backing{memory};
    auto space = std::make_shared<Tegra::MemoryManager>(system, backing, 32, 1ULL << 32, 16, 12);
    const std::weak_ptr weak_space{space};
    const NvCore::SessionId owner{7};
    auto channels = std::make_unique<NvCore::NativeGpuChannels>();
    std::shared_ptr<NvCore::NativeGpuChannel> retained;
    std::weak_ptr<NvCore::NativeGpuSchedulingGroup> weak_group;
    {
        Devices::nvhost_tsg_gpu device{system, *channels};
        device.OnOpen(owner, 3);
        REQUIRE(channels->Allocate(4, owner, 0) == NvResult::Success);
        REQUIRE(channels->Allocate(5, NvCore::SessionId{8}, 0) == NvResult::Success);
        retained = channels->Get(4, owner);
        REQUIRE(retained->BindAddressSpace(space) == NvResult::Success);
        std::array<u8, 24> input{};
        std::array<u8, 32> output{};
        output.fill(0xcc);
        const auto set_word = [&](size_t offset, u32 value) {
            const u32_le wire{value};
            std::memcpy(input.data() + offset, &wire, sizeof(wire));
        };
        const Ioctl bind{.raw = 0xc018540b};
        for (u32 fd : {0U, 3U, 5U, 0xffffffffU}) {
            set_word(0, fd);
            REQUIRE(device.Ioctl1(3, bind, input, output) == NvResult::BadParameter);
        }
        set_word(0, 4);
        set_word(4, 64);
        REQUIRE(device.Ioctl1(3, bind, input, output) == NvResult::BadParameter);
        set_word(4, 1);
        REQUIRE(device.Ioctl1(3, bind, std::span{input}.first(23), output) ==
                NvResult::InvalidSize);
        REQUIRE(device.Ioctl1(3, bind, input, std::span{output}.first(23)) ==
                NvResult::InvalidSize);
        REQUIRE(device.Ioctl1(3, Ioctl{.raw = 0x4018540b}, input, output) ==
                NvResult::BadParameter);
        REQUIRE(retained->GetGroup() == nullptr);
        REQUIRE(output[0] == 0xcc);
        input[8] = 0xaa;
        REQUIRE(device.Ioctl1(3, bind, input, output) == NvResult::Success);
        REQUIRE(output[8] == 0xaa);
        REQUIRE(output[24] == 0xcc);
        REQUIRE(retained->GetSubcontext() == 1U);
        REQUIRE(device.Ioctl1(3, bind, input, output) == NvResult::InvalidState);
        REQUIRE(device.Ioctl1(3, Ioctl{.raw = 0xc0045402}, input, output) == NvResult::Success);
        REQUIRE(retained->GetGroup() == nullptr);
        REQUIRE(device.Ioctl1(3, Ioctl{.raw = 0xc0045402}, input, output) ==
                NvResult::BadParameter);
        REQUIRE(device.Ioctl1(3, Ioctl{.raw = 0xc0045401}, input, output) == NvResult::Success);
        REQUIRE(retained->GetSubcontext() == 0U);
        weak_group = retained->GetGroup();
        device.OnClose(3);
        REQUIRE_FALSE(device.GetSchedulingParameters());
        REQUIRE_FALSE(weak_group.expired());
        space.reset();
        REQUIRE_FALSE(weak_space.expired());
    }
    channels.reset();
    REQUIRE_FALSE(retained->IsOpen());
    REQUIRE(retained->BindAddressSpace(retained->GetAddressSpace()) == NvResult::InvalidState);
    REQUIRE_FALSE(weak_group.expired());
    REQUIRE_FALSE(weak_space.expired());
    retained.reset();
    REQUIRE(weak_group.expired());
    REQUIRE(weak_space.expired());
}
