// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstring>
#include <span>
#include "common/common_types.h"
#include "common/swap.h"
#include "video_core/host1x/syncpoint_manager.h"

namespace Tegra::Host1x {

class SyncpointShim {
public:
    static constexpr u32 Stride = 0x10000;
    static constexpr u32 Count = SyncpointManager::NativeSyncpointCount;
    static constexpr u64 Size = u64{Stride} * Count;

    explicit SyncpointShim(SyncpointManager& manager) : m_manager{manager} {}

    static bool IsValidRead(u64 offset, size_t size) {
        if (offset >= Size || size > Size - offset) {
            return false;
        }
        if (size == 0) {
            return true;
        }
        const auto register_offset = offset % Stride;
        // Only the counter register is implemented. Other registers in each slot are not RAM.
        return register_offset < sizeof(u32) && size <= sizeof(u32) - register_offset;
    }

    bool Read(u64 offset, std::span<u8> output) const {
        if (!IsValidRead(offset, output.size())) {
            return false;
        }
        if (output.empty()) {
            return true;
        }
        const auto register_offset = offset % Stride;
        const u32_le value{m_manager.GetHostSyncpointValue(static_cast<u32>(offset / Stride))};
        std::memcpy(output.data(), reinterpret_cast<const u8*>(&value) + register_offset,
                    output.size());
        return true;
    }

private:
    SyncpointManager& m_manager;
};

} // namespace Tegra::Host1x
