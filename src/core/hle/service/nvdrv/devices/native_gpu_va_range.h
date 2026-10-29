// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <limits>
#include <optional>
#include "common/alignment.h"
#include "common/common_types.h"

namespace Service::Nvidia::Devices {

struct NativeGpuVaRange {
    u64 first;
    u64 end;

    static std::optional<NativeGpuVaRange> Create(u64 partition_first, u64 partition_last,
                                                  u64 requested_first, u64 requested_end) {
        if (partition_first > partition_last ||
            partition_last == (std::numeric_limits<u64>::max)()) {
            return std::nullopt;
        }
        const u64 first = requested_first ? requested_first : partition_first;
        const u64 end = requested_end ? requested_end : partition_last + 1;
        if (first < partition_first || first >= end || end > partition_last + 1 ||
            !Common::IsAligned(first, 0x1000) || !Common::IsAligned(end, 0x1000)) {
            return std::nullopt;
        }
        return NativeGpuVaRange{first, end};
    }
};

} // namespace Service::Nvidia::Devices
