// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <optional>
#include "video_core/control/native_color_clear.h"

namespace Tegra::Control {

struct NativeRenderTarget {
    NativeColorClear color;
    std::array<u32, 10> registers;
    u32 padded_size_bytes;
    bool single_gob_volume;
};

struct NativeRenderTargets {
    using Registers = std::array<std::array<u32, 10>, 8>;

    std::array<NativeRenderTarget, 8> targets;
    u32 count;

    // C797 CT_SELECT identity routing and equally sized, equally formatted
    // color targets. Volume normalization is limited to the first single GOB;
    // padded_size_bytes preserves the original tile size; only color.size_bytes
    // belongs to the selected view and requires mapped backing.
    static std::optional<NativeRenderTargets> Decode(u32 control, u32 srgb_write,
                                                     const Registers& registers);
};

} // namespace Tegra::Control
