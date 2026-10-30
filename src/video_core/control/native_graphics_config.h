// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/common_types.h"

namespace Tegra::Control {

// Modeled graphics-context settings, never host GPU register addresses.
class NativeGraphicsConfig {
public:
    struct Host {
        void* context{};
        bool (*wait_for_idle)(void*){};
        bool (*should_stop)(void*){};
    };
    enum class Error {
        None,
        UnsupportedRegister,
        UnsupportedMask,
        UnsupportedValue,
        WaitFailed,
        Cancelled
    };

    Error ApplyMaskedRegister(u32 address, u32 value, u32 mask, const Host& host);
    bool PolygonOffsetNanIsZero() const;
    bool HelperInvocationLoadsEnabled() const;
    u32 TextureSamplerDebug() const;
    u32 TextureSamplerDebugKnownMask() const;
    u32 TextureLodDebug() const;
    u32 TextureLodDebugKnownMask() const;

private:
    bool polygon_offset_nan_is_zero{};
    bool helper_invocation_loads_enabled{};
    u32 texture_sampler_debug{};
    u32 texture_sampler_debug_known_mask{};
    u32 texture_lod_debug{};
    u32 texture_lod_debug_known_mask{};
};

} // namespace Tegra::Control
