// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <utility>
#include <catch2/catch_test_macros.hpp>
#include "video_core/control/native_render_targets.h"

namespace {
using Targets = Tegra::Control::NativeRenderTargets;
using Clear = Tegra::Control::NativeColorClear;

u32 IdentityControl(u32 count) {
    u32 control = count;
    for (u32 index = 0; index < 8; ++index) {
        control |= index << (4 + index * 3);
    }
    return control;
}

Targets::Registers ColorRegisters() {
    Targets::Registers registers{};
    for (u32 index = 0; index < 8; ++index) {
        registers[index] = {
            0x21, 0x10000000U + index * 0x100000U, 640, 360, 0xe0, 0x40, 1, 0x56e00, 0, 0};
    }
    return registers;
}

Targets::Registers VolumeRegisters(u32 block_depth = 3) {
    Targets::Registers registers{};
    for (u32 index = 0; index < 8; ++index) {
        registers[index] = {0x22, 0x634bb000U + index * 512,
                            16,   8,
                            0xe0, 0x10000U | (block_depth << 8),
                            1,    0x400,
                            0,    0};
    }
    return registers;
}
} // namespace

TEST_CASE("Native render targets preserve regular color footprints",
          "[native-shader][video_core]") {
    const auto registers = ColorRegisters();
    for (u32 count = 1; count <= 8; ++count) {
        const auto targets = Targets::Decode(IdentityControl(count), 0, registers);
        REQUIRE(targets);
        REQUIRE(targets->count == count);
        for (u32 index = 0; index < count; ++index) {
            Clear::Registers expected{};
            expected.argument = 0x3c;
            expected.target = registers[index];
            const auto clear = Clear::Decode(expected);
            REQUIRE(clear);
            const auto& target = targets->targets[index];
            REQUIRE(target.registers == registers[index]);
            REQUIRE_FALSE(target.single_gob_volume);
            REQUIRE(target.padded_size_bytes == clear->size_bytes);
            REQUIRE(target.color.address == clear->address);
            REQUIRE(target.color.format == clear->format);
            REQUIRE(target.color.width == clear->width);
            REQUIRE(target.color.height == clear->height);
            REQUIRE(target.color.block_height == clear->block_height);
            REQUIRE(target.color.size_bytes == clear->size_bytes);
            REQUIRE(target.color.extent_x == clear->extent_x);
            REQUIRE(target.color.extent_y == clear->extent_y);
        }
    }
}

TEST_CASE("Native target format identity distinguishes UNORM from sRGB",
          "[native-shader][video_core]") {
    auto registers = ColorRegisters();
    for (auto& target : registers)
        target[4] = 0xd5;
    for (u32 srgb_write : {0U, 1U}) {
        const auto targets = Targets::Decode(IdentityControl(8), srgb_write, registers);
        REQUIRE(targets);
        for (const auto& target : targets->targets) {
            REQUIRE(target.color.format == Clear::Format::RGBA8Unorm);
            REQUIRE(target.registers[4] == 0xd5);
            REQUIRE(target.color.size_bytes == 0xf0000);
        }
    }
    REQUIRE_FALSE(Targets::Decode(IdentityControl(8), 2, registers));
    registers[1][4] = 0xd6;
    REQUIRE_FALSE(Targets::Decode(IdentityControl(8), 1, registers));
    registers[1][4] = 0xe0;
    REQUIRE_FALSE(Targets::Decode(IdentityControl(8), 1, registers));
}

TEST_CASE("Native MRT volume slices touch disjoint GOBs despite padded overlap",
          "[native-shader][video_core]") {
    const auto registers = VolumeRegisters();
    REQUIRE(IdentityControl(8) == 0x0fac6888);
    const auto targets = Targets::Decode(IdentityControl(8), 0, registers);
    REQUIRE(targets);
    REQUIRE(targets->count == 8);
    for (u32 index = 0; index < 8; ++index) {
        const auto& target = targets->targets[index];
        REQUIRE(target.registers == registers[index]);
        REQUIRE(target.single_gob_volume);
        REQUIRE(target.padded_size_bytes == 4096);
        REQUIRE(target.color.address == 0x22634bb000ULL + index * 512);
        REQUIRE(target.color.size_bytes == 512);
        REQUIRE(target.color.width == 16);
        REQUIRE(target.color.height == 8);
        REQUIRE(target.color.block_height == 0);
        if (index != 0) {
            const auto& prior = targets->targets[index - 1];
            REQUIRE(prior.color.address + prior.color.size_bytes == target.color.address);
            REQUIRE(prior.color.address + prior.padded_size_bytes > target.color.address);
        }
    }
    for (u32 block_depth = 0; block_depth <= 5; ++block_depth) {
        auto small = VolumeRegisters(block_depth);
        small[0][2] = 1;
        small[0][3] = 1;
        const auto decoded = Targets::Decode(IdentityControl(1), 0, small);
        REQUIRE(decoded);
        REQUIRE(decoded->targets[0].color.size_bytes == 512);
        REQUIRE(decoded->targets[0].padded_size_bytes == (512U << block_depth));
    }
}

TEST_CASE("Native render target routing rejects invalid active state",
          "[native-shader][video_core]") {
    const auto registers = VolumeRegisters();
    for (u32 count : {0U, 9U, 15U}) {
        REQUIRE_FALSE(Targets::Decode(IdentityControl(count), 0, registers));
    }
    for (u32 bit = 28; bit < 32; ++bit) {
        REQUIRE_FALSE(Targets::Decode(IdentityControl(8) | (1U << bit), 0, registers));
    }
    for (u32 index = 0; index < 8; ++index) {
        REQUIRE_FALSE(Targets::Decode(IdentityControl(8) ^ (1U << (4 + index * 3)), 0, registers));
    }
    // Inactive target contents and their otherwise valid routing fields do not
    // select resources; only the active prefix belongs to this draw.
    auto inactive = registers;
    inactive[7].fill(0xffffffff);
    REQUIRE(Targets::Decode(IdentityControl(1) ^ (7U << 25), 0, inactive));
}

TEST_CASE("Native render targets reject unsupported volume shapes and target fields",
          "[native-shader][video_core]") {
    for (const auto [word, value] : std::array<std::pair<u32, u32>, 18>{{{0, 0x100},
                                                                         {1, 0x634bb001},
                                                                         {2, 0},
                                                                         {2, 17},
                                                                         {3, 0},
                                                                         {3, 9},
                                                                         {4, 0xd7},
                                                                         {5, 0x10600},
                                                                         {5, 0x11300},
                                                                         {5, 0x10301},
                                                                         {5, 0x10310},
                                                                         {5, 0x30300},
                                                                         {5, 0x300},
                                                                         {6, 0},
                                                                         {6, 2},
                                                                         {8, 1},
                                                                         {9, 1},
                                                                         {2, 0x10000010}}}) {
        auto registers = VolumeRegisters();
        registers[0][word] = value;
        REQUIRE_FALSE(Targets::Decode(IdentityControl(1), 0, registers));
    }
}

TEST_CASE("Native MRT requires matching formats and extents without touched aliases",
          "[native-shader][video_core]") {
    for (const auto [word, value] :
         std::array<std::pair<u32, u32>, 3>{{{2, 8}, {3, 4}, {4, 0xd6}}}) {
        auto registers = VolumeRegisters();
        registers[1][word] = value;
        REQUIRE_FALSE(Targets::Decode(IdentityControl(2), 1, registers));
    }
    auto duplicate = VolumeRegisters();
    duplicate[1][1] = duplicate[0][1];
    REQUIRE_FALSE(Targets::Decode(IdentityControl(2), 0, duplicate));
    auto overlap = ColorRegisters();
    overlap[1][1] = overlap[0][1] + 512;
    REQUIRE_FALSE(Targets::Decode(IdentityControl(2), 0, overlap));
    overlap[1][1] = overlap[0][1] + 0xf0000;
    REQUIRE(Targets::Decode(IdentityControl(2), 0, overlap));
    auto reversed = VolumeRegisters();
    std::swap(reversed[0], reversed[1]);
    REQUIRE(Targets::Decode(IdentityControl(2), 0, reversed));
}

TEST_CASE("Native render target padding does not extend the selected mapping requirement",
          "[native-shader][video_core]") {
    auto registers = VolumeRegisters();
    registers[0][0] = 0xff;
    registers[0][1] = 0xfffff000;
    const auto boundary = Targets::Decode(IdentityControl(1), 0, registers);
    REQUIRE(boundary);
    REQUIRE(boundary->targets[0].color.address + boundary->targets[0].padded_size_bytes ==
            (u64{1} << 40));
    registers[0][1] = 0xfffffe00;
    const auto slice = Targets::Decode(IdentityControl(1), 0, registers);
    REQUIRE(slice);
    REQUIRE(slice->targets[0].padded_size_bytes == 4096);
    REQUIRE(slice->targets[0].color.address + slice->targets[0].color.size_bytes == (u64{1} << 40));
    registers[0][5] = 0;
    const auto regular = Targets::Decode(IdentityControl(1), 0, registers);
    REQUIRE(regular);
    REQUIRE(regular->targets[0].padded_size_bytes == 512);
    REQUIRE(regular->targets[0].color.address + regular->targets[0].padded_size_bytes ==
            (u64{1} << 40));
}

TEST_CASE("Native sRGB render targets require explicit write conversion",
          "[native-shader][video_core]") {
    for (bool volume : {false, true}) {
        auto registers = volume ? VolumeRegisters() : ColorRegisters();
        for (auto& target : registers) {
            target[4] = 0xd6;
        }
        const auto enabled = Targets::Decode(IdentityControl(8), 1, registers);
        REQUIRE(enabled);
        REQUIRE(enabled->targets[0].color.format == Clear::Format::RGBA8Srgb);
        REQUIRE_FALSE(Targets::Decode(IdentityControl(8), 0, registers));
        REQUIRE_FALSE(Targets::Decode(IdentityControl(8), 2, registers));
    }
}
