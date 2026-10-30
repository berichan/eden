// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "video_core/control/native_graphics_config.h"

namespace Tegra::Control {
namespace {
struct BundlePreset {
    u32 address;
    u32 value;
    u32 known_mask;
    u32 field_bits;
};
// NVIDIA ga10b_gr_init_get_access_map names SWDX_TC_BUNDLE_CTRL/ADDR.
// deko3d's graphics initialization establishes these nibble/halfword presets.
// They are opaque context state: no host register access or guest GPUVA decoding.
constexpr std::array<BundlePreset, 8> BundlePresets{{
    {0x418e40, 0x77777777, 0xffffffff, 4},
    {0x418e44, 0x00000777, 0x00000fff, 4},
    {0x418e58, 0x04f10842, 0xffffffff, 16},
    {0x418e5c, 0x00e90053, 0xffffffff, 16},
    {0x418e60, 0x00eb00ea, 0xffffffff, 16},
    {0x418e64, 0x02090208, 0xffffffff, 16},
    {0x418e68, 0x020b020a, 0xffffffff, 16},
    {0x418e6c, 0x00000644, 0x0000ffff, 16},
}};

std::optional<size_t> BundleIndex(u32 address) {
    for (size_t index = 0; index < BundlePresets.size(); ++index) {
        if (BundlePresets[index].address == address) {
            return index;
        }
    }
    return std::nullopt;
}
} // namespace

NativeGraphicsConfig::Error NativeGraphicsConfig::ApplyMaskedRegister(u32 address, u32 value,
                                                                      u32 mask, const Host& host) {
    // NVIDIA GA10B hw_gr_ga10b.h: gr_pri_gpcs_setup_debug, bit0.
    // gv11b_gr_init_set_default_gfx_regs enables this field for graphics contexts.
    const bool setup_debug = address == 0x418800;
    const bool sampler_debug = address == 0x419a08;
    const bool sm_dispatch_control = address == 0x419ba4;
    const bool lod_debug = address == 0x419a04;
    const auto bundle_index = BundleIndex(address);
    if (!setup_debug && !sampler_debug && !sm_dispatch_control && !lod_debug && !bundle_index) {
        return Error::UnsupportedRegister;
    }
    const u32 expected_mask = setup_debug ? 1U : (sampler_debug ? 0x10U : 8U);
    const bool conservative_setup = setup_debug && mask == 0x01800000;
    if (bundle_index) {
        const auto& preset = BundlePresets[*bundle_index];
        bool valid_mask{};
        for (u32 shift = 0; shift < 32; shift += preset.field_bits) {
            const u32 field_mask = ((1U << preset.field_bits) - 1) << shift;
            valid_mask |= mask == field_mask && (mask & preset.known_mask) == mask;
        }
        if (!valid_mask) {
            return Error::UnsupportedMask;
        }
        if (value != (preset.value & mask)) {
            return Error::UnsupportedValue;
        }
    } else if (!conservative_setup &&
               ((lod_debug && mask != 1 && mask != 2) || (!lod_debug && mask != expected_mask))) {
        return Error::UnsupportedMask;
    }
    const bool valid_value = conservative_setup ? value == 0
                             : lod_debug        ? value == mask
                             : setup_debug      ? value <= 1
                                                : value == 0;
    if (!bundle_index && !valid_value) {
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
    if (bundle_index) {
        // Preserve the exact hardware initialization presets. The synchronous
        // draw adapter reads explicit method state instead of replaying bundle transport.
        auto& bundle = swdx_bundles[*bundle_index];
        bundle.value = (bundle.value & ~mask) | value;
        bundle.known_mask |= mask;
    } else if (setup_debug) {
        // Conservative-raster initialization clears bits23/24 independently of bit0.
        // The bounded draw adapter rejects conservative raster and depth bias;
        // those effects need to consume these policies when implemented.
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
        // unresolved, so native texture operations remain unsupported.
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

std::optional<NativeGraphicsConfig::MaskedValue> NativeGraphicsConfig::SwdxBundle(
    u32 address) const {
    const auto index = BundleIndex(address);
    return index ? std::optional{swdx_bundles[*index]} : std::nullopt;
}

} // namespace Tegra::Control
