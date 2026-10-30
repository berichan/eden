// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "video_core/control/native_graphics_config.h"

namespace Tegra::Control {

NativeGraphicsConfig::Error NativeGraphicsConfig::ApplyMaskedRegister(u32 address, u32 value,
                                                                      u32 mask, const Host& host) {
    // NVIDIA GA10B hw_gr_ga10b.h: gr_pri_gpcs_setup_debug, bit0.
    // gv11b_gr_init_set_default_gfx_regs enables this field for graphics contexts.
    const bool setup_debug = address == 0x418800;
    const bool sampler_debug = address == 0x419a08;
    const bool sm_dispatch_control = address == 0x419ba4;
    const bool lod_debug = address == 0x419a04;
    if (!setup_debug && !sampler_debug && !sm_dispatch_control && !lod_debug) {
        return Error::UnsupportedRegister;
    }
    const u32 expected_mask = setup_debug ? 1U : (sampler_debug ? 0x10U : 8U);
    const bool conservative_setup = setup_debug && mask == 0x01800000;
    if (!conservative_setup &&
        ((lod_debug && mask != 1 && mask != 2) || (!lod_debug && mask != expected_mask))) {
        return Error::UnsupportedMask;
    }
    const bool valid_value = conservative_setup ? value == 0
                             : lod_debug        ? value == mask
                             : setup_debug      ? value <= 1
                                                : value == 0;
    if (!valid_value) {
        return Error::UnsupportedValue;
    }
    if (host.should_stop && host.should_stop(host.context)) {
        return Error::Cancelled;
    }
    if (!host.wait_for_idle || !host.wait_for_idle(host.context)) {
        return Error::WaitFailed;
    }
    if (host.should_stop && host.should_stop(host.context)) {
        return Error::Cancelled;
    }
    if (setup_debug) {
        // Conservative-raster initialization clears bits23/24 independently of bit0.
        // Draws remain guarded until the renderer implements these context policies.
        setup_debug_value = (setup_debug_value & ~mask) | value;
        setup_debug_known_mask |= mask;
    } else if (sm_dispatch_control) {
        // GA10B SM_DISP_CTRL.KILLED_LD_IS_NOP: clearing bit3 enables helper loads.
        helper_invocation_loads_enabled = true;
    } else if (lod_debug) {
        // TEX_LOD_DBG initialization: bit1 is CUBESEAM_ANISO, bit0 unresolved.
        // Record only the separate presets; native texture operations remain guarded.
        texture_lod_debug = (texture_lod_debug & ~mask) | value;
        texture_lod_debug_known_mask |= mask;
    } else {
        // NVIDIA names this register TEX_SAMP_DBG; deko3d initializes bit4 to zero.
        // Only that exact initialization is recorded. Its rendering meaning remains
        // unresolved, so native textures and draw actions must remain unsupported.
        texture_sampler_debug = (texture_sampler_debug & ~mask) | value;
        texture_sampler_debug_known_mask |= mask;
    }
    return Error::None;
}

bool NativeGraphicsConfig::PolygonOffsetNanIsZero() const {
    return (setup_debug_value & 1) != 0;
}

u32 NativeGraphicsConfig::SetupDebug() const {
    return setup_debug_value;
}

u32 NativeGraphicsConfig::SetupDebugKnownMask() const {
    return setup_debug_known_mask;
}

bool NativeGraphicsConfig::HelperInvocationLoadsEnabled() const {
    return helper_invocation_loads_enabled;
}

u32 NativeGraphicsConfig::TextureSamplerDebug() const {
    return texture_sampler_debug;
}

u32 NativeGraphicsConfig::TextureSamplerDebugKnownMask() const {
    return texture_sampler_debug_known_mask;
}

u32 NativeGraphicsConfig::TextureLodDebug() const {
    return texture_lod_debug;
}

u32 NativeGraphicsConfig::TextureLodDebugKnownMask() const {
    return texture_lod_debug_known_mask;
}

} // namespace Tegra::Control
