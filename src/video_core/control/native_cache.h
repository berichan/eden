// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <optional>
#include "common/common_types.h"

namespace Tegra::Control {

struct NativeCacheInvalidation {
    bool instructions{};
    bool global_data{};
    bool constants{};
    bool texture_data{};
    bool texture_headers{};
    bool samplers{};
};

std::optional<NativeCacheInvalidation> DecodeNativeGraphicsCacheAction(u32 method, u32 value);

} // namespace Tegra::Control
