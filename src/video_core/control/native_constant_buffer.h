// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <span>
#include "common/common_types.h"

namespace Tegra::Control {

class NativeConstantBuffer {
public:
    struct Selection {
        u32 size;
        u32 address_upper;
        u32 address_lower;
        u32 offset;
    };
    struct Host {
        void* context;
        bool (*is_writable)(void*, GPUVAddr, size_t);
        bool (*wait_for_idle)(void*);
        bool (*write_checked)(void*, GPUVAddr, std::span<const u32>);
        bool (*should_stop)(void*);
    };
    enum class Error { None, InvalidSelection, OutOfBounds, MemoryFault, WaitFailed, Cancelled };

    static Error Upload(Selection& selection, std::span<const u32> words, const Host& host);
};

} // namespace Tegra::Control
