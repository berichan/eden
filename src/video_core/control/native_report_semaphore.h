// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include "common/common_types.h"

namespace Tegra::Control {

class NativeReportSemaphore {
public:
    enum class Engine { Graphics, Compute };
    struct Host {
        void* context{};
        bool (*is_writable)(void*, GPUVAddr, size_t){};
        bool (*wait_for_idle)(void*){};
        bool (*write_checked)(void*, GPUVAddr, u32){};
        bool (*should_stop)(void*){};
    };
    struct TimestampHost {
        Host common;
        u64 (*timestamp_ns)(void*){};
        bool (*write_checked)(void*, GPUVAddr, u64 timestamp){};
    };
    enum class Error {
        None,
        UnsupportedOperation,
        InvalidAddress,
        MemoryFault,
        WaitFailed,
        Cancelled
    };

    static Error Release(u32 upper, u32 lower, u32 payload, u32 operation, const Host& host,
                         Engine engine = Engine::Graphics);
    static Error ReportTimestamp(u32 upper, u32 lower, u32 payload, u32 operation,
                                 const TimestampHost& host);
};

} // namespace Tegra::Control
