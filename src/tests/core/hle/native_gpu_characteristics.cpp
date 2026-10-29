// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <catch2/catch_test_macros.hpp>
#include "core/hle/service/nvdrv/devices/native_gpu_characteristics.h"

namespace {

template <typename T>
T Read(std::span<const u8> bytes, size_t offset) {
    T value{};
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    return value;
}

} // namespace

TEST_CASE("Native GPU characteristics expose the native ABI and Ampere identity", "[ipc][nvdrv]") {
    std::array<u8, 0xe0> input{};
    std::array<u8, 0xe0> output{};
    std::array<u8, 0xd8> inline_output{};
    const u64_le requested{0xd8};
    std::memcpy(input.data(), &requested, sizeof(requested));
    std::fill(input.begin() + 8, input.end(), 0x5a);
    REQUIRE(Service::Nvidia::Devices::ReadNativeGpuCharacteristics(input, output, inline_output) ==
            Service::Nvidia::NvResult::Success);
    REQUIRE(Read<u64_le>(output, 0) == 0xd8ULL);
    REQUIRE(std::equal(input.begin() + 8, input.end(), output.begin() + 8));
    REQUIRE(Read<u32_le>(inline_output, 0) == 0x170U);
    REQUIRE(Read<u32_le>(inline_output, 4) == 0xeU);
    REQUIRE(Read<u32_le>(inline_output, 0x20) == 6U);
    REQUIRE(Read<u32_le>(inline_output, 0x3c) == 0x808U);
    REQUIRE(Read<u32_le>(inline_output, 0x5c) == 0xc797U);
    REQUIRE(Read<u32_le>(inline_output, 0x60) == 0xc7c0U);
    REQUIRE(Read<s16_le>(inline_output, 0x70) == 0x1f);
    REQUIRE(Read<s16_le>(inline_output, 0x76) == 0x21);
    REQUIRE(Read<s16_le>(inline_output, 0x78) == 0xf);
    REQUIRE(Read<s16_le>(inline_output, 0x7a) == -1);
    REQUIRE(Read<u32_le>(inline_output, 0x98) == 1U);
    const std::array<u8, 6> name{'g', 'a', '1', '0', 'f', 0};
    REQUIRE(std::equal(name.begin(), name.end(), inline_output.begin() + 0xa4));
    REQUIRE(Read<u64_le>(inline_output, 0xc8) == 0ULL);
}

TEST_CASE("Native characteristics copies respect both size limits and allow size-only queries",
          "[ipc][nvdrv]") {
    std::array<u8, 0xe0> input{};
    std::array<u8, 0xe8> output{};
    std::array<u8, 0xe8> inline_output{};
    const Service::Nvidia::Devices::NativeGpuCharacteristics expected{};
    const auto* expected_bytes = reinterpret_cast<const u8*>(&expected);
    for (const u64 requested :
         std::array<u64, 6>{0, 1, 0x77, 0xd0, 0xd8, std::numeric_limits<u64>::max()}) {
        for (const size_t available : {0U, 2U, 0x78U, 0xd0U, 0xe0U}) {
            CAPTURE(requested, available);
            const u64_le encoded{requested};
            std::memcpy(input.data(), &encoded, sizeof(encoded));
            output.fill(0xcc);
            inline_output.fill(0xcc);
            REQUIRE(Service::Nvidia::Devices::ReadNativeGpuCharacteristics(
                        input, std::span{output}.first(0xe0),
                        std::span{inline_output}.subspan(4, available)) ==
                    Service::Nvidia::NvResult::Success);
            const auto copied = std::min<u64>({requested, available, sizeof(expected)});
            REQUIRE(std::equal(expected_bytes, expected_bytes + copied, inline_output.begin() + 4));
            REQUIRE(std::all_of(inline_output.begin(), inline_output.begin() + 4,
                                [](u8 value) { return value == 0xcc; }));
            REQUIRE(std::all_of(inline_output.begin() + 4 + copied, inline_output.end(),
                                [](u8 value) { return value == 0xcc; }));
            REQUIRE(Read<u64_le>(output, 0) == sizeof(expected));
            REQUIRE(std::all_of(output.begin() + 0xe0, output.end(),
                                [](u8 value) { return value == 0xcc; }));
        }
    }
}

TEST_CASE("Native characteristics reject truncated fixed buffers before writing", "[ipc][nvdrv]") {
    std::array<u8, 0xe0> input{};
    std::array<u8, 0xe0> output{};
    std::array<u8, 0xd8> inline_output{};
    for (const size_t size : {0U, 7U, 8U, 0xb0U, 0xdfU}) {
        CAPTURE(size);
        output.fill(0xcc);
        inline_output.fill(0xdd);
        REQUIRE(Service::Nvidia::Devices::ReadNativeGpuCharacteristics(std::span{input}.first(size),
                                                                       output, inline_output) ==
                Service::Nvidia::NvResult::InvalidSize);
        REQUIRE(Service::Nvidia::Devices::ReadNativeGpuCharacteristics(
                    input, std::span{output}.first(size), inline_output) ==
                Service::Nvidia::NvResult::InvalidSize);
        REQUIRE(std::all_of(output.begin(), output.end(), [](u8 value) { return value == 0xcc; }));
        REQUIRE(std::all_of(inline_output.begin(), inline_output.end(),
                            [](u8 value) { return value == 0xdd; }));
    }
}

TEST_CASE("Native engine enumeration reports total size and copies only whole records",
          "[ipc][nvdrv]") {
    std::array<u8, 0x10> input{};
    std::array<u8, 0x18> output{};
    std::array<u8, 0x38> inline_output{};
    input.fill(0x5a);
    for (const size_t available : {0U, 1U, 15U, 16U, 17U, 31U, 32U, 48U}) {
        CAPTURE(available);
        output.fill(0xcc);
        inline_output.fill(0xdd);
        REQUIRE(Service::Nvidia::Devices::ReadNativeGpuEngineInfo(
                    input, std::span{output}.first(0x10),
                    std::span{inline_output}.subspan(4, available)) ==
                Service::Nvidia::NvResult::Success);
        REQUIRE(Read<u32_le>(output, 0) == 32U);
        REQUIRE(std::equal(input.begin() + 4, input.end(), output.begin() + 4));
        REQUIRE(std::all_of(output.begin() + 0x10, output.end(),
                            [](u8 value) { return value == 0xcc; }));
        const auto count = std::min<size_t>(available / 16, 2);
        if (count >= 1) {
            REQUIRE(Read<u32_le>(inline_output, 4) == 1U);
            REQUIRE(Read<u32_le>(inline_output, 8) == 0U);
            REQUIRE(Read<s32_le>(inline_output, 12) == 0);
            REQUIRE(Read<u32_le>(inline_output, 16) == 0U);
        }
        if (count == 2) {
            REQUIRE(Read<u32_le>(inline_output, 20) == 2U);
            REQUIRE(Read<u32_le>(inline_output, 24) == 0U);
            REQUIRE(Read<s32_le>(inline_output, 28) == 1);
            REQUIRE(Read<u32_le>(inline_output, 32) == 0U);
        }
        REQUIRE(std::all_of(inline_output.begin(), inline_output.begin() + 4,
                            [](u8 value) { return value == 0xdd; }));
        REQUIRE(std::all_of(inline_output.begin() + 4 + count * 16, inline_output.end(),
                            [](u8 value) { return value == 0xdd; }));
    }
    REQUIRE(Service::Nvidia::Devices::ReadNativeGpuEngineInfo(std::span{input}.first(8), output,
                                                              inline_output) ==
            Service::Nvidia::NvResult::InvalidSize);
    REQUIRE(Service::Nvidia::Devices::ReadNativeGpuEngineInfo(input, std::span{output}.first(8),
                                                              inline_output) ==
            Service::Nvidia::NvResult::InvalidSize);
}
