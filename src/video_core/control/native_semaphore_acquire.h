// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include "common/common_types.h"

namespace Tegra::Control {
class NativeSemaphoreAcquire {
public:
    struct Host {
        void* context{};
        bool (*is_readable)(void*, GPUVAddr, size_t){};
        bool (*wait_for_idle)(void*){};
        bool (*read_checked)(void*, GPUVAddr, u32&){};
        bool (*should_stop)(void*){};
    };
    enum class Error {
        None,
        UnsupportedOperation,
        InvalidAddress,
        MemoryFault,
        WaitFailed,
        Cancelled,
        NotReady
    };
    static constexpr bool IsSatisfied(u32 value, u32 payload) {
        return u32(value - payload) < 0x80000000U;
    }
    // Unsatisfied waits need resumable channel scheduling, not a blocking GPU-worker loop.
    static Error Check(u32 upper, u32 lower, u32 payload, u32 operation, const Host& host);
};
} // namespace Tegra::Control
