// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <limits>
#include "video_core/control/native_color_clear.h"

namespace Tegra::Control {

std::optional<NativeColorClear> NativeColorClear::Decode(const Registers& regs) {
    // COLOR_TARGET A..LAYER and CLEAR_CONTROL: NVIDIA classes/3d/clc797.h.
    const u32 target_index = (regs.argument >> 6) & 15;
    if ((regs.argument & 0x3f) != 0x3c || (regs.argument & ~0x3fcU) || target_index >= 8 ||
        regs.target[0] > 0xff || (regs.target[1] & 0x1ff) || (regs.target[2] & 0xf0000000U) ||
        (regs.target[3] & 0xfffe0000U) || regs.target[2] == 0 || regs.target[3] == 0 ||
        (regs.target[4] != 0xe0 && regs.target[4] != 0xd6) || (regs.target[5] & ~0xf0U) ||
        ((regs.target[5] >> 4) & 15) > 5 || regs.target[6] != 1 || regs.target[8] != 0 ||
        regs.target[9] != 0 || (regs.control & ~0x1111U) || regs.scissor_enable > 1 ||
        regs.anti_alias_raster != 0 || regs.anti_alias_enable != 0 || regs.anti_alias != 0) {
        return std::nullopt;
    }
    const auto format = regs.target[4] == 0xe0 ? Format::B10G11R11Float : Format::RGBA8Srgb;
    if (format == Format::RGBA8Srgb) {
        // NVK supplies float clears unchanged to this sRGB target format. An
        // explicit SRGB_WRITE disable has unverified clear semantics; RGB
        // endpoints are invariant under either encoding, so remain supported.
        if (regs.srgb_write > 1 ||
            (regs.srgb_write == 0 &&
             !std::all_of(regs.color.begin(), regs.color.begin() + 3,
                          [](u32 value) { return value == 0 || value == 0x3f800000; })))
            return std::nullopt;
    }
    const auto address = (GPUVAddr{regs.target[0]} << 32) | regs.target[1];
    const auto width = regs.target[2];
    const auto height = regs.target[3];
    const auto block_height = (regs.target[5] >> 4) & 15;
    const u64 pitch = (u64{width} * 4 + 63) & ~u64{63};
    const u64 tile_height = u64{8} << block_height;
    const u64 aligned_height = (height + tile_height - 1) & ~(tile_height - 1);
    const u64 size = pitch * aligned_height;
    // ARRAY_PITCH is spacing between layers, in DWORDs. It does not bound the
    // sole layer at index zero, including when a target binds one image mip.
    if (size > (std::numeric_limits<u32>::max)() || size > (u64{1} << 40) - address) {
        return std::nullopt;
    }

    u32 x0{}, y0{}, x1{width}, y1{height};
    const auto intersect = [&](u32 horizontal, u32 vertical, bool origin_size) {
        const u32 left = horizontal & 0xffff;
        const u32 top = vertical & 0xffff;
        const u32 right = origin_size ? left + (horizontal >> 16) : horizontal >> 16;
        const u32 bottom = origin_size ? top + (vertical >> 16) : vertical >> 16;
        x0 = (std::max)(x0, (std::min)(left, width));
        y0 = (std::max)(y0, (std::min)(top, height));
        x1 = (std::min)(x1, right);
        y1 = (std::min)(y1, bottom);
    };
    if (regs.control & 0x10)
        intersect(regs.rect_x, regs.rect_y, false);
    if ((regs.control & 0x100) && regs.scissor_enable)
        intersect(regs.scissor_x, regs.scissor_y, false);
    if (regs.control & 0x1000)
        intersect(regs.viewport_x, regs.viewport_y, true);
    return NativeColorClear{address,
                            format,
                            width,
                            height,
                            block_height,
                            static_cast<u32>(size),
                            x0,
                            y0,
                            x1 > x0 ? x1 - x0 : 0,
                            y1 > y0 ? y1 - y0 : 0,
                            regs.color};
}

} // namespace Tegra::Control
