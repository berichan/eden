// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <optional>
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
    u32 SetupDebug() const;
    u32 SetupDebugKnownMask() const;
    bool HelperInvocationLoadsEnabled() const;
    u32 TextureSamplerDebug() const;
    u32 TextureSamplerDebugKnownMask() const;
    u32 TextureLodDebug() const;
    u32 TextureLodDebugKnownMask() const;
    struct MaskedValue {
        u32 value{};
        u32 known_mask{};
    };
    std::optional<MaskedValue> SwdxBundle(u32 address) const;

private:
    u32 setup_debug_value{};
    u32 setup_debug_known_mask{};
    bool helper_invocation_loads_enabled{};
    u32 texture_sampler_debug{};
    u32 texture_sampler_debug_known_mask{};
    u32 texture_lod_debug{};
    u32 texture_lod_debug_known_mask{};
    std::array<MaskedValue, 8> swdx_bundles{};
};

} // namespace Tegra::Control
