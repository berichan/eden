// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstring>
#include <catch2/catch_test_macros.hpp>
#include "common/scope_exit.h"
#include "common/settings.h"
#include "common/swap.h"
#include "core/core.h"
#include "core/device_memory.h"
#include "core/hle/kernel/k_event.h"
#include "core/hle/kernel/k_resource_limit.h"
#include "core/hle/kernel/k_scheduler.h"
#include "core/hle/kernel/kernel.h"
#include "core/hle/service/nvdrv/core/native_gpu_channel.h"
#include "core/hle/service/nvdrv/core/native_gpu_error_notifier.h"
#include "core/hle/service/nvdrv/devices/nvhost_native_gpu.h"
#include "video_core/control/native_gpu_context.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/memory_manager.h"

TEST_CASE("Native graphics context requests allocate owned class state with a valid binding",
          "[core][ipc]") {
    using namespace Service::Nvidia;
    Core::System system;
    Core::DeviceMemory memory;
    Tegra::MaxwellDeviceMemoryManager backing{memory};
    auto space = std::make_shared<Tegra::MemoryManager>(system, backing, 32, 1ULL << 32, 16, 12);
    Tegra::Host1x::SyncpointManager syncpoints;
    NvCore::NativeGpuChannels channels{syncpoints};
    constexpr NvCore::SessionId owner{7};
    Devices::nvhost_native_gpu device{system, channels};
    device.OnOpen(owner, 4);
    const Ioctl allocate{.raw = 0xc0104809};
    std::array<u8, 16> input{};
    std::array<u8, 24> output{};
    const auto set_word = [&](size_t offset, u32 value) {
        const u32_le wire{value};
        std::memcpy(input.data() + offset, &wire, sizeof(wire));
    };
    set_word(0, 0xc797);
    set_word(8, 0xdeadbeef);
    set_word(12, 0x12345678);
    output.fill(0xcc);
    REQUIRE(device.Ioctl1(5, allocate, input, output) == NvResult::InvalidState);
    REQUIRE(device.Ioctl1(4, allocate, input, output) == NvResult::BadParameter);
    REQUIRE(channels.Allocate(4, owner, 0) == NvResult::Success);
    const auto channel = channels.Get(4, owner);
    for (u32 raw : {0x40104809U, 0x80104809U, 0x00104809U}) {
        REQUIRE(device.Ioctl1(4, Ioctl{.raw = raw}, input, output) == NvResult::BadParameter);
    }
    for (u32 raw : {0xc00c4809U, 0xc0144809U}) {
        REQUIRE(device.Ioctl1(4, Ioctl{.raw = raw}, input, output) == NvResult::InvalidSize);
    }
    REQUIRE(device.Ioctl1(4, allocate, std::span{input}.first(15), output) ==
            NvResult::InvalidSize);
    REQUIRE(device.Ioctl1(4, allocate, input, std::span{output}.first(15)) ==
            NvResult::InvalidSize);
    REQUIRE(device.Ioctl1(4, allocate, input, output) == NvResult::InvalidState);
    REQUIRE(channel->BindAddressSpace(space) == NvResult::Success);
    REQUIRE(device.Ioctl1(4, allocate, input, output) == NvResult::InvalidState);
    auto group = std::make_shared<NvCore::NativeGpuSchedulingGroup>(owner);
    REQUIRE(channel->BindGroup(group, 0) == NvResult::Success);
    REQUIRE(channels.AllocateObjectContext(4, NvCore::SessionId{8}, 0xc797, 0) ==
            NvResult::BadParameter);
    for (u32 class_id : {0U, 0xdeadU, 0xffffffffU}) {
        set_word(0, class_id);
        REQUIRE(device.Ioctl1(4, allocate, input, output) == NvResult::BadParameter);
    }
    for (u32 class_id : {0x902dU, 0xa140U, 0xc7c0U, 0xc76fU, 0xc7b5U}) {
        set_word(0, class_id);
        REQUIRE(device.Ioctl1(4, allocate, input, output) == NvResult::NotSupported);
    }
    set_word(0, 0xc797);
    for (u32 flags : {1U, 2U, 4U, 0xffffffffU}) {
        set_word(4, flags);
        REQUIRE(device.Ioctl1(4, allocate, input, output) == NvResult::NotSupported);
    }
    REQUIRE(channel->GetGraphicsContext() == nullptr);
    REQUIRE(output.front() == 0xcc);
    REQUIRE(output.back() == 0xcc);
    set_word(4, 0);
    REQUIRE(device.Ioctl1(4, allocate, input, output) == NvResult::Success);
    REQUIRE(std::memcmp(input.data(), output.data(), input.size()) == 0);
    REQUIRE(output[16] == 0xcc);
    auto context = channel->GetGraphicsContext();
    REQUIRE(context != nullptr);
    REQUIRE(context->GetAddressSpace() == space);
    REQUIRE(context->GetState()->registers[0] == 0xc797);
    REQUIRE(context->GetState()->shadow_registers[0] == 0xc797);
    REQUIRE(context->GetState()->registers[0x202] == 0);
    context->GetState()->registers[0x202] = 2560;
    REQUIRE(device.Ioctl1(4, allocate, input, input) == NvResult::Success);
    REQUIRE(channel->GetGraphicsContext() == context);
    REQUIRE(context->GetState()->registers[0x202] == 2560);
    REQUIRE(device.Ioctl2(4, allocate, input, input, output) == NvResult::NotSupported);
    REQUIRE(device.Ioctl3(4, allocate, input, output, output) == NvResult::NotSupported);
    REQUIRE(channel->UnbindGroup(group) == NvResult::Success);
    REQUIRE(channel->GetGraphicsContext() == nullptr);
    REQUIRE(device.Ioctl1(4, allocate, input, output) == NvResult::InvalidState);
    REQUIRE(channel->BindGroup(group, 0) == NvResult::Success);
    REQUIRE(device.Ioctl1(4, allocate, input, output) == NvResult::Success);
    REQUIRE(channel->GetGraphicsContext() == context);
    device.OnClose(4);
    REQUIRE(device.Ioctl1(4, allocate, input, output) == NvResult::InvalidState);
    REQUIRE(context->GetState()->registers[0x202] == 2560);
    REQUIRE(syncpoints.GetHostSyncpointValue(channel->GetChannelIndex() + 17) == 0);
}

TEST_CASE("Native graphics state follows scheduling groups and retains address space bindings",
          "[core][ipc]") {
    using namespace Service::Nvidia;
    Core::System system;
    Core::DeviceMemory memory;
    Tegra::MaxwellDeviceMemoryManager backing{memory};
    auto first_space =
        std::make_shared<Tegra::MemoryManager>(system, backing, 32, 1ULL << 32, 16, 12);
    auto second_space =
        std::make_shared<Tegra::MemoryManager>(system, backing, 32, 1ULL << 32, 16, 12);
    constexpr NvCore::SessionId owner{7};
    Tegra::Host1x::SyncpointManager syncpoints;
    std::shared_ptr<const Tegra::Control::NativeGpuGraphicsContext> retained;
    std::weak_ptr<Tegra::Control::AmpereGraphicsState> weak_state;
    std::weak_ptr<Tegra::MemoryManager> weak_space = first_space;
    {
        NvCore::NativeGpuChannels channels{syncpoints};
        auto group = std::make_shared<NvCore::NativeGpuSchedulingGroup>(owner);
        auto separate_group = std::make_shared<NvCore::NativeGpuSchedulingGroup>(owner);
        for (DeviceFD fd : {4, 5, 6, 7}) {
            REQUIRE(channels.Allocate(fd, owner, 0) == NvResult::Success);
            const auto channel = channels.Get(fd, owner);
            REQUIRE(channel->BindAddressSpace(fd == 6 ? second_space : first_space) ==
                    NvResult::Success);
            REQUIRE(channel->BindGroup(fd == 7 ? separate_group : group, fd == 6 ? 1 : 0) ==
                    NvResult::Success);
            REQUIRE(channels.AllocateObjectContext(fd, owner, 0xc797, 0) == NvResult::Success);
        }
        retained = channels.Get(4, owner)->GetGraphicsContext();
        weak_state = retained->GetState();
        REQUIRE(channels.Get(5, owner)->GetGraphicsContext() == retained);
        const auto alternate = channels.Get(6, owner)->GetGraphicsContext();
        REQUIRE(alternate != retained);
        REQUIRE(alternate->GetState() == retained->GetState());
        REQUIRE(alternate->GetAddressSpace() == second_space);
        REQUIRE(channels.Get(7, owner)->GetGraphicsContext()->GetState() != retained->GetState());
        retained->GetState()->registers[0x202] = 2560;
        REQUIRE(alternate->GetState()->registers[0x202] == 2560);
        REQUIRE(channels.Get(7, owner)->GetGraphicsContext()->GetState()->registers[0x202] == 0);
        REQUIRE(channels.Allocate(8, owner, 1) == NvResult::Success);
        const auto copy = channels.Get(8, owner);
        auto copy_group = std::make_shared<NvCore::NativeGpuSchedulingGroup>(owner);
        REQUIRE(copy->BindAddressSpace(first_space) == NvResult::Success);
        REQUIRE(copy->BindGroup(copy_group, 0) == NvResult::Success);
        REQUIRE(channels.AllocateObjectContext(8, owner, 0xc797, 0) == NvResult::BadParameter);
        REQUIRE(copy->GetGraphicsContext() == nullptr);
        first_space.reset();
        second_space.reset();
    }
    REQUIRE_FALSE(weak_state.expired());
    REQUIRE_FALSE(weak_space.expired());
    REQUIRE(retained->GetState()->registers[0x202] == 2560);
    retained.reset();
    REQUIRE(weak_state.expired());
    REQUIRE(weak_space.expired());
}

TEST_CASE("Native channel client data is opaque, independent, and limited to its owner",
          "[core][ipc]") {
    using namespace Service::Nvidia;
    Core::System system;
    Tegra::Host1x::SyncpointManager syncpoints;
    NvCore::NativeGpuChannels channels{syncpoints};
    constexpr NvCore::SessionId owner{7};
    Devices::nvhost_native_gpu first{system, channels};
    Devices::nvhost_native_gpu second{system, channels};
    first.OnOpen(owner, 4);
    second.OnOpen(owner, 5);
    REQUIRE(channels.Allocate(4, owner, 0) == NvResult::Success);
    REQUIRE(channels.Allocate(5, owner, 0) == NvResult::Success);
    const Ioctl set_data{.raw = 0x40084714};
    const Ioctl get_data{.raw = 0x80084715};
    const u64_le value{0xfedcba9876543210};
    std::array<u8, 8> input;
    std::memcpy(input.data(), &value, sizeof(value));
    std::array<u8, 16> output;
    output.fill(0xcc);
    REQUIRE(first.Ioctl1(4, get_data, {}, output) == NvResult::Success);
    u64_le observed;
    std::memcpy(&observed, output.data(), sizeof(observed));
    REQUIRE(observed == 0);
    for (u32 raw : {0x80084714U, 0xc0084714U, 0x00084714U, 0x40084715U, 0xc0084715U, 0x00084715U}) {
        REQUIRE(first.Ioctl1(4, Ioctl{.raw = raw}, input, output) == NvResult::BadParameter);
    }
    REQUIRE(first.Ioctl1(4, Ioctl{.raw = 0x40044714}, input, output) == NvResult::InvalidSize);
    REQUIRE(first.Ioctl1(4, set_data, std::span{input}.first(7), output) == NvResult::InvalidSize);
    REQUIRE(first.Ioctl1(4, get_data, {}, std::span{output}.first(7)) == NvResult::InvalidSize);
    REQUIRE(channels.SetClientData(4, NvCore::SessionId{8}, value) == NvResult::BadParameter);
    output.fill(0xcc);
    REQUIRE(first.Ioctl1(4, set_data, input, {}) == NvResult::Success);
    REQUIRE(first.Ioctl1(4, get_data, {}, output) == NvResult::Success);
    std::memcpy(&observed, output.data(), sizeof(observed));
    REQUIRE(observed == value);
    REQUIRE(output[8] == 0xcc);
    REQUIRE(second.Ioctl1(5, get_data, {}, output) == NvResult::Success);
    std::memcpy(&observed, output.data(), sizeof(observed));
    REQUIRE(observed == 0);
    u64 untouched{0x1234};
    REQUIRE(channels.GetClientData(&untouched, 4, NvCore::SessionId{8}) == NvResult::BadParameter);
    REQUIRE(untouched == 0x1234);
    first.OnClose(4);
    REQUIRE(first.Ioctl1(4, get_data, {}, output) == NvResult::InvalidState);
    first.OnOpen(owner, 4);
    REQUIRE(channels.Allocate(4, owner, 0) == NvResult::Success);
    REQUIRE(channels.GetClientData(&untouched, 4, owner) == NvResult::Success);
    REQUIRE(untouched == 0);
}

TEST_CASE("Native Z-cull binding is owned by shared graphics state and preserves its first address",
          "[core][ipc]") {
    using namespace Service::Nvidia;
    Core::System system;
    Core::DeviceMemory memory;
    Tegra::MaxwellDeviceMemoryManager backing{memory};
    auto space = std::make_shared<Tegra::MemoryManager>(system, backing, 32, 1ULL << 32, 16, 12);
    Tegra::Host1x::SyncpointManager syncpoints;
    NvCore::NativeGpuChannels channels{syncpoints};
    constexpr NvCore::SessionId owner{7};
    Devices::nvhost_native_gpu device{system, channels};
    device.OnOpen(owner, 4);
    const Ioctl bind{.raw = 0xc010480b};
    std::array<u8, 16> input{};
    std::array<u8, 24> output{};
    output.fill(0xcc);
    const auto set_params = [&](u64 address, u32 mode) {
        const u64_le wire_address{address};
        const u32_le wire_mode{mode};
        std::memcpy(input.data(), &wire_address, sizeof(wire_address));
        std::memcpy(input.data() + 8, &wire_mode, sizeof(wire_mode));
        input[12] = 0xa5;
    };
    REQUIRE(device.Ioctl1(4, bind, input, output) == NvResult::BadParameter);
    REQUIRE(channels.Allocate(4, owner, 0) == NvResult::Success);
    const auto channel = channels.Get(4, owner);
    REQUIRE(device.Ioctl1(4, bind, input, output) == NvResult::InvalidState);
    REQUIRE(channel->BindAddressSpace(space) == NvResult::Success);
    REQUIRE(device.Ioctl1(4, bind, input, output) == NvResult::InvalidState);
    auto group = std::make_shared<NvCore::NativeGpuSchedulingGroup>(owner);
    REQUIRE(channel->BindGroup(group, 0) == NvResult::Success);
    for (u32 raw : {0x4010480bU, 0x8010480bU, 0x0010480bU}) {
        REQUIRE(device.Ioctl1(4, Ioctl{.raw = raw}, input, output) == NvResult::BadParameter);
    }
    REQUIRE(device.Ioctl1(4, Ioctl{.raw = 0xc014480b}, input, output) == NvResult::InvalidSize);
    REQUIRE(device.Ioctl1(4, bind, std::span{input}.first(15), output) == NvResult::InvalidSize);
    REQUIRE(device.Ioctl1(4, bind, input, std::span{output}.first(15)) == NvResult::InvalidSize);
    set_params(0x20000000, 4);
    REQUIRE(device.Ioctl1(4, bind, input, output) == NvResult::BadParameter);
    set_params(0, 2);
    REQUIRE(device.Ioctl1(4, bind, input, output) == NvResult::BadParameter);
    REQUIRE(channel->GetGraphicsContext() == nullptr);
    REQUIRE(output[0] == 0xcc);
    REQUIRE(channels.BindZCull(4, NvCore::SessionId{8}, 0, 0) == NvResult::BadParameter);
    for (u32 mode : {0U, 1U, 3U}) {
        set_params(0, mode);
        REQUIRE(device.Ioctl1(4, bind, input, output) == NvResult::Success);
        REQUIRE(channel->GetGraphicsContext()->GetZCullBinding()->address == 0);
        REQUIRE(channel->GetGraphicsContext()->GetZCullBinding()->mode == mode);
    }
    REQUIRE(channels.AllocateObjectContext(4, owner, 0xc797, 0) == NvResult::Success);
    set_params(0x20000000, 2);
    REQUIRE(device.Ioctl1(4, bind, input, output) == NvResult::Success);
    REQUIRE(std::memcmp(input.data(), output.data(), input.size()) == 0);
    REQUIRE(output[16] == 0xcc);
    auto context = channel->GetGraphicsContext();
    REQUIRE(context->GetZCullBinding()->address == 0x20000000);
    REQUIRE(context->GetZCullBinding()->mode == 2);
    set_params(0x30000000, 1);
    REQUIRE(device.Ioctl1(4, bind, input, input) == NvResult::Success);
    REQUIRE(context->GetZCullBinding()->address == 0x20000000);
    REQUIRE(context->GetZCullBinding()->mode == 2);
    REQUIRE(channels.Allocate(5, owner, 0) == NvResult::Success);
    const auto sibling = channels.Get(5, owner);
    REQUIRE(sibling->BindAddressSpace(space) == NvResult::Success);
    REQUIRE(sibling->BindGroup(group, 0) == NvResult::Success);
    REQUIRE(channels.AllocateObjectContext(5, owner, 0xc797, 0) == NvResult::Success);
    REQUIRE(sibling->GetGraphicsContext()->GetZCullBinding()->address == 0x20000000);
    REQUIRE(channel->UnbindGroup(group) == NvResult::Success);
    REQUIRE(channel->BindGroup(group, 0) == NvResult::Success);
    REQUIRE(channels.AllocateObjectContext(4, owner, 0xc797, 0) == NvResult::Success);
    REQUIRE(channel->GetGraphicsContext()->GetZCullBinding()->address == 0x20000000);
    device.OnClose(4);
    REQUIRE(context->GetZCullBinding()->address == 0x20000000);
    REQUIRE(device.Ioctl1(4, bind, input, output) == NvResult::InvalidState);
}

TEST_CASE("Native error notification records latch the first fault and reset on configuration",
          "[core][ipc]") {
    using namespace Service::Nvidia;
    Core::System system;
    Tegra::Host1x::SyncpointManager syncpoints;
    NvCore::NativeGpuChannels channels{syncpoints};
    constexpr NvCore::SessionId owner{7};
    Devices::nvhost_native_gpu device{system, channels};
    device.OnOpen(owner, 4);
    REQUIRE(channels.Allocate(4, owner, 0) == NvResult::Success);
    const auto notifier = channels.Get(4, owner)->GetErrorNotifier();
    const Ioctl configure{.raw = 0xc018480c};
    const Ioctl read{.raw = 0xc0104817};
    std::array<u8, 24> input;
    input.fill(0xa5);
    std::array<u8, 32> output;
    output.fill(0xcc);
    REQUIRE_FALSE(notifier->IsEnabled());
    REQUIRE(notifier->Read().status == 0);
    for (u32 raw : {0x4018480cU, 0x8018480cU, 0x0018480cU, 0x40104817U, 0x80104817U, 0x00104817U}) {
        REQUIRE(device.Ioctl1(4, Ioctl{.raw = raw}, input, output) == NvResult::BadParameter);
    }
    REQUIRE(device.Ioctl1(4, configure, std::span{input}.first(23), output) ==
            NvResult::InvalidSize);
    REQUIRE(device.Ioctl1(4, read, input, std::span{output}.first(15)) == NvResult::InvalidSize);
    REQUIRE(output[0] == 0xcc);
    REQUIRE(device.Ioctl1(4, configure, input, output) == NvResult::Success);
    REQUIRE(notifier->IsEnabled());
    REQUIRE(std::memcmp(input.data(), output.data(), input.size()) == 0);
    REQUIRE(output[24] == 0xcc);
    REQUIRE(notifier->RecordError(0x12345678, 0x1122334455667788));
    REQUIRE_FALSE(notifier->RecordError(0x99999999, 0xaabbccdd));
    REQUIRE(device.Ioctl1(4, read, input, output) == NvResult::Success);
    u64_le timestamp;
    u32_le information;
    std::array<u16_le, 2> details;
    std::memcpy(&timestamp, output.data(), sizeof(timestamp));
    std::memcpy(&information, output.data() + 8, sizeof(information));
    std::memcpy(details.data(), output.data() + 12, sizeof(details));
    REQUIRE(timestamp == 0x1122334455667788);
    REQUIRE(information == 0x12345678);
    REQUIRE(details[0] == 0);
    REQUIRE(details[1] == 0xffff);
    REQUIRE(notifier->Read().status == 0xffff);
    REQUIRE(output[16] == input[16]);
    std::fill(input.begin() + 16, input.begin() + 20, 0);
    REQUIRE(device.Ioctl1(4, configure, input, output) == NvResult::Success);
    REQUIRE_FALSE(notifier->IsEnabled());
    REQUIRE(notifier->Read().timestamp == 0);
    REQUIRE(notifier->Read().status == 0);
    REQUIRE(notifier->RecordError(0x87654321, 0x1234));
    REQUIRE(notifier->Read().information == 0x87654321);
    REQUIRE(device.QueryEvent(3) == nullptr);
    device.OnClose(4);
    REQUIRE(device.QueryEvent(7) == nullptr);
}

TEST_CASE("Native channel timeout requests retain the complete configured budget", "[core][ipc]") {
    using namespace Service::Nvidia;
    Core::System system;
    Tegra::Host1x::SyncpointManager syncpoints;
    NvCore::NativeGpuChannels channels{syncpoints};
    constexpr NvCore::SessionId owner{7};
    Devices::nvhost_native_gpu device{system, channels};
    device.OnOpen(owner, 4);
    REQUIRE(channels.Allocate(4, owner, 0) == NvResult::Success);
    const auto channel = channels.Get(4, owner);
    REQUIRE_FALSE(channel->GetTimeout());
    std::array<u8, 4> input{};
    const Ioctl set_timeout{.raw = 0x40044803};
    for (u32 raw : {0xc0044803U, 0x80044803U, 0x00044803U}) {
        REQUIRE(device.Ioctl1(4, Ioctl{.raw = raw}, input, {}) == NvResult::BadParameter);
    }
    REQUIRE(device.Ioctl1(4, Ioctl{.raw = 0x40084803}, input, {}) == NvResult::InvalidSize);
    REQUIRE(device.Ioctl1(4, set_timeout, std::span{input}.first(3), {}) == NvResult::InvalidSize);
    REQUIRE(channels.SetTimeout(4, NvCore::SessionId{8}, 1234) == NvResult::BadParameter);
    REQUIRE_FALSE(channel->GetTimeout());
    for (u32 budget : {0U, 1000U, 0xffffffffU}) {
        const u32_le wire{budget};
        std::memcpy(input.data(), &wire, sizeof(wire));
        REQUIRE(device.Ioctl1(4, set_timeout, input, {}) == NvResult::Success);
        REQUIRE(channel->GetTimeout() == budget);
    }
    device.OnClose(4);
    REQUIRE(device.Ioctl1(4, set_timeout, input, {}) == NvResult::InvalidState);
}

TEST_CASE("Native error events signal real faults and survive descriptor closure with consumers",
          "[core][ipc]") {
    using namespace Service::Nvidia;
    const bool multicore = Settings::values.use_multi_core.GetValue();
    Settings::values.use_multi_core = false;
    SCOPE_EXIT {
        Settings::values.use_multi_core = multicore;
    };
    Core::System system;
    system.Initialize();
    auto& kernel = system.Kernel();
    kernel.Initialize();
    SCOPE_EXIT {
        kernel.Shutdown();
    };
    auto* limits = kernel.GetSystemResourceLimit();
    const auto events_before = limits->GetCurrentValue(Kernel::LimitableResource::EventCountMax);
    Tegra::Host1x::SyncpointManager syncpoints;
    NvCore::NativeGpuChannels channels{syncpoints};
    constexpr NvCore::SessionId owner{7};
    std::shared_ptr<NvCore::NativeGpuErrorNotifier> retained;
    Kernel::KReadableEvent* readable;
    {
        Devices::nvhost_native_gpu device{system, channels};
        device.OnOpen(owner, 4);
        REQUIRE(channels.Allocate(4, owner, 0) == NvResult::Success);
        retained = channels.Get(4, owner)->GetErrorNotifier();
        retained->Configure(true);
        REQUIRE(retained->RecordError(0x1234, 0x5678));
        auto* event = device.QueryEvent(7);
        REQUIRE(event != nullptr);
        REQUIRE(device.QueryEvent(7) == event);
        REQUIRE(limits->GetCurrentValue(Kernel::LimitableResource::EventCountMax) ==
                events_before + 1);
        readable = &event->GetReadableEvent();
        readable->Open(kernel);
        {
            Kernel::KScopedSchedulerLock lock{kernel};
            REQUIRE(readable->IsSignaled(kernel));
        }
        REQUIRE(readable->Reset(kernel).IsSuccess());
        retained->Configure(false);
        REQUIRE(retained->RecordError(0x2222, 0x3333));
        {
            Kernel::KScopedSchedulerLock lock{kernel};
            REQUIRE_FALSE(readable->IsSignaled(kernel));
        }
        retained->Configure(true);
        REQUIRE(retained->RecordError(0x4444, 0x5555));
        {
            Kernel::KScopedSchedulerLock lock{kernel};
            REQUIRE(readable->IsSignaled(kernel));
        }
        device.OnClose(4);
        REQUIRE(device.QueryEvent(7) == nullptr);
    }
    REQUIRE(retained->Read().information == 0x4444);
    retained.reset();
    REQUIRE(limits->GetCurrentValue(Kernel::LimitableResource::EventCountMax) == events_before + 1);
    REQUIRE(readable->Reset(kernel).IsSuccess());
    readable->Close(kernel);
    REQUIRE(limits->GetCurrentValue(Kernel::LimitableResource::EventCountMax) == events_before);
}
