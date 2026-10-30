// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <optional>
#include "common/common_types.h"

namespace Tegra::Control {

struct NativeColorClear {
    enum class Format { B10G11R11Float, RGBA8Srgb };
    struct Registers {
        u32 argument;
        std::array<u32, 10> target;
        std::array<u32, 4> color;
        u32 control;
        u32 rect_x, rect_y;
        u32 scissor_enable, scissor_x, scissor_y;
        u32 viewport_x, viewport_y;
        u32 anti_alias_raster, anti_alias_enable, anti_alias;
        u32 srgb_write;
    };

    GPUVAddr address;
    Format format;
    u32 width, height;
    u32 block_height;
    u32 size_bytes;
    u32 x, y, extent_x, extent_y;
    std::array<u32, 4> color;

    // C797 single-layer 2D BF10GF11RF11 or A8BL8GL8RL8, full color channels.
    // All extents and the complete tiled allocation are checked before host use.
    static std::optional<NativeColorClear> Decode(const Registers& regs);
    bool Empty() const {
        return extent_x == 0 || extent_y == 0;
    }
};

} // namespace Tegra::Control
