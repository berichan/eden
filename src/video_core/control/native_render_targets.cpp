// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "video_core/control/native_render_targets.h"

namespace Tegra::Control {

std::optional<NativeRenderTargets> NativeRenderTargets::Decode(u32 control, u32 srgb_write,
                                                               const Registers& registers) {
    const u32 count = control & 15;
    if (count == 0 || count > 8 || (control & 0xf0000000U)) {
        return std::nullopt;
    }

    NativeRenderTargets result{};
    result.count = count;
    for (u32 index = 0; index < count; ++index) {
        if (((control >> (4 + index * 3)) & 7) != index) {
            return std::nullopt;
        }
        const auto& raw = registers[index];
        NativeColorClear::Registers clear_registers{};
        clear_registers.argument = 0x3c;
        clear_registers.target = raw;
        clear_registers.srgb_write = srgb_write;

        const u32 block_depth = (raw[5] >> 8) & 15;
        const bool single_gob_volume = raw[5] == (0x10000U | (block_depth << 8));
        if (single_gob_volume) {
            if (block_depth > 5 || raw[2] == 0 || raw[2] > 16 || raw[3] == 0 || raw[3] > 8) {
                return std::nullopt;
            }
            // NIL's volume address formula reduces z=0 to the first GOB when
            // X and Y both fit one GOB. Only this selected window is rendered
            // and copied back; neighboring slices must not be downloaded.
            clear_registers.target[5] = 0;
        }
        const auto color = NativeColorClear::Decode(clear_registers);
        if (!color || (color->format == NativeColorClear::Format::RGBA8Srgb && srgb_write != 1) ||
            (color->format == NativeColorClear::Format::RGBA8Unorm && srgb_write != 0)) {
            return std::nullopt;
        }
        const u32 padded_size = single_gob_volume ? 512U << block_depth : color->size_bytes;
        if (index != 0) {
            const auto& first = result.targets[0].color;
            if (color->format != first.format || color->width != first.width ||
                color->height != first.height) {
                return std::nullopt;
            }
        }
        for (u32 prior = 0; prior < index; ++prior) {
            const auto& other = result.targets[prior].color;
            if (color->address < other.address + other.size_bytes &&
                other.address < color->address + color->size_bytes) {
                return std::nullopt;
            }
        }
        result.targets[index] = {*color, raw, padded_size, single_gob_volume};
    }
    return result;
}

} // namespace Tegra::Control
