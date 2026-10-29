// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstring>
#include <catch2/catch_test_macros.hpp>
#include "common/swap.h"
#include "core/core.h"
#include "core/hle/service/nvdrv/devices/nvhost_tsg_gpu.h"

TEST_CASE("Native GPU scheduling groups validate configuration and close independently",
          "[core][ipc]") {
    using namespace Service::Nvidia;
    Core::System system;
    Devices::nvhost_tsg_gpu first{system};
    Devices::nvhost_tsg_gpu second{system};
    constexpr DeviceFD FirstFd = 3;
    constexpr DeviceFD SecondFd = 4;
    const Ioctl interleave{.raw = 0xc0085407};
    const Ioctl promotion{.raw = 0xc008540d};
    std::array<u8, 12> input{};
    std::array<u8, 12> output{};
    input[4] = 0xaa;
    const auto set_value = [&](u32 value) {
        const u32_le wire_value{value};
        std::memcpy(input.data(), &wire_value, sizeof(wire_value));
    };
    REQUIRE_FALSE(first.GetSchedulingParameters());
    REQUIRE(first.Ioctl1(FirstFd, interleave, input, output) == NvResult::InvalidState);
    first.OnOpen(NvCore::SessionId{0}, FirstFd);
    second.OnOpen(NvCore::SessionId{0}, SecondFd);
    REQUIRE(first.GetSchedulingParameters()->interleave_level == 0);
    for (u32 level = 0; level <= 2; ++level) {
        set_value(level);
        output.fill(0xcc);
        REQUIRE(first.Ioctl1(FirstFd, interleave, input, output) == NvResult::Success);
        REQUIRE(first.GetSchedulingParameters()->interleave_level == level);
        REQUIRE(output[0] == level);
        REQUIRE(output[4] == 0xaa);
        REQUIRE(output[8] == 0xcc);
        REQUIRE(second.GetSchedulingParameters()->interleave_level == 0);
    }
    for (u32 invalid : {3U, 0xffffffffU}) {
        set_value(invalid);
        output.fill(0xcc);
        REQUIRE(first.Ioctl1(FirstFd, interleave, input, output) == NvResult::BadParameter);
        REQUIRE(first.GetSchedulingParameters()->interleave_level == 2);
        REQUIRE(output[0] == 0xcc);
    }
    for (u32 value : {1U, 2U, 4U}) {
        set_value(value);
        REQUIRE(first.Ioctl1(FirstFd, promotion, input, output) == NvResult::Success);
        REQUIRE(first.GetSchedulingParameters()->l2_promotion == value);
    }
    for (u32 invalid : {0U, 3U, 5U, 0xffffffffU}) {
        set_value(invalid);
        REQUIRE(first.Ioctl1(FirstFd, promotion, input, output) == NvResult::BadParameter);
        REQUIRE(first.GetSchedulingParameters()->l2_promotion == 4);
    }

    set_value(1);
    REQUIRE(first.Ioctl1(SecondFd, interleave, input, output) == NvResult::InvalidState);
    for (u32 raw : {0x00085407U, 0x40085407U, 0x80085407U}) {
        REQUIRE(first.Ioctl1(FirstFd, Ioctl{.raw = raw}, input, output) == NvResult::BadParameter);
    }
    for (u32 raw : {0xc0045407U, 0xc00c5407U}) {
        REQUIRE(first.Ioctl1(FirstFd, Ioctl{.raw = raw}, input, output) == NvResult::InvalidSize);
    }
    for (size_t length : {size_t{0}, size_t{4}, size_t{7}}) {
        REQUIRE(first.Ioctl1(FirstFd, interleave, std::span{input}.first(length), output) ==
                NvResult::InvalidSize);
        REQUIRE(first.Ioctl1(FirstFd, interleave, input, std::span{output}.first(length)) ==
                NvResult::InvalidSize);
    }
    REQUIRE(first.GetSchedulingParameters()->interleave_level == 2);
    REQUIRE(first.GetSchedulingParameters()->l2_promotion == 4);
    for (u32 raw : {0xc0045401U, 0xc018540bU, 0x00005403U, 0x00005404U, 0x00005405U, 0xc0045408U,
                    0xc0084807U, 0xc00854ffU}) {
        REQUIRE(first.Ioctl1(FirstFd, Ioctl{.raw = raw}, input, output) == NvResult::NotSupported);
    }
    REQUIRE(first.Ioctl2(FirstFd, interleave, input, input, output) == NvResult::NotSupported);
    REQUIRE(first.Ioctl3(FirstFd, interleave, input, output, output) == NvResult::NotSupported);
    REQUIRE(first.Ioctl1(FirstFd, interleave, input, input) == NvResult::Success);
    REQUIRE(first.GetSchedulingParameters()->interleave_level == 1);
    first.OnClose(SecondFd);
    REQUIRE(first.GetSchedulingParameters().has_value());
    first.OnClose(FirstFd);
    REQUIRE_FALSE(first.GetSchedulingParameters());
    REQUIRE(first.Ioctl1(FirstFd, interleave, input, output) == NvResult::InvalidState);
    REQUIRE(first.Ioctl2(FirstFd, interleave, input, input, output) == NvResult::InvalidState);
    REQUIRE(first.Ioctl3(FirstFd, interleave, input, output, output) == NvResult::InvalidState);
    REQUIRE(second.GetSchedulingParameters().has_value());
    first.OnOpen(NvCore::SessionId{1}, 5);
    REQUIRE(first.GetSchedulingParameters()->interleave_level == 0);
    REQUIRE(first.GetSchedulingParameters()->l2_promotion == 0);
    REQUIRE(first.Ioctl1(FirstFd, interleave, input, output) == NvResult::InvalidState);
    first.OnClose(5);
    second.OnClose(SecondFd);
}
