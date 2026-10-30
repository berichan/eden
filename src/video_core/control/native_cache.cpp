// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "video_core/control/native_cache.h"

namespace Tegra::Control {

std::optional<NativeCacheInvalidation> DecodeNativeGraphicsCacheAction(u32 method, u32 value) {
    // Tagged/single-line invalidation is unsupported. The backend must provide
    // real ordering for these NVIDIA C797 NO_WFI methods and the SDK variant below.
    NativeCacheInvalidation action;
    switch (method) {
    case 0x369: // INVALIDATE_SHADER_CACHES_NO_WFI
        if (value & ~0x1011U) {
            return std::nullopt;
        }
        action.instructions = (value & 1) != 0;
        action.global_data = (value & 0x10) != 0;
        action.constants = (value & 0x1000) != 0;
        return action;
    case 0x3dd: // SDK ordered texture-data invalidation; not named in NVIDIA's public class header.
    case 0x4a2: // INVALIDATE_TEXTURE_DATA_CACHE_NO_WFI
        action.texture_data = true;
        break;
    case 0x50a: // INVALIDATE_TEXTURE_HEADER_CACHE_NO_WFI
        action.texture_headers = true;
        break;
    case 0x509: // INVALIDATE_SAMPLER_CACHE_NO_WFI
        action.samplers = true;
        break;
    default:
        return std::nullopt;
    }
    return value == 0 ? std::optional{action} : std::nullopt;
}

} // namespace Tegra::Control
