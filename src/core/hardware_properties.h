// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <tuple>

#include "common/bit_util.h"
#include "common/common_types.h"

namespace Core {

namespace Hardware {

constexpr u64 BASE_CLOCK_RATE = 1'020'000'000; // Default CPU Frequency = 1020 MHz
constexpr u64 CNTFREQ = 19'200'000;            // CNTPCT_EL0 Frequency = 19.2 MHz
// Capacity for guest application cores. Keep in sync with Dynarmic's exclusive monitor.
constexpr u32 MAX_CPU_CORES = 6;

enum class CpuProfile {
    Nx,
    Ounce,
};

constexpr u32 GetCounterFrequency(CpuProfile profile) {
    return profile == CpuProfile::Ounce ? 31'250'000U : static_cast<u32>(CNTFREQ);
}

constexpr u32 GetCpuCoreCount(CpuProfile profile) {
    return profile == CpuProfile::Ounce ? 6 : 4;
}

// Virtual to Physical core map.
constexpr std::array<s32, Common::BitSize<u64>()> VirtualToPhysicalCoreMap{
    0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3,
};

static constexpr inline size_t NumVirtualCores = Common::BitSize<u64>();

static constexpr inline u64 VirtualCoreMask = [] {
    u64 mask = 0;
    for (size_t i = 0; i < NumVirtualCores; ++i) {
        mask |= (UINT64_C(1) << i);
    }
    return mask;
}();

constexpr s32 VirtualToPhysicalCore(CpuProfile profile, s32 core) {
    return profile == CpuProfile::Ounce && core < 6 ? core : VirtualToPhysicalCoreMap[core];
}

constexpr u64 GetVirtualCoreMask(CpuProfile profile) {
    return profile == CpuProfile::Ounce ? (1ULL << GetCpuCoreCount(profile)) - 1 : VirtualCoreMask;
}

static constexpr inline u64 ConvertVirtualCoreMaskToPhysical(CpuProfile profile, u64 v_core_mask) {
    u64 p_core_mask = 0;
    while (v_core_mask != 0) {
        const u64 next = std::countr_zero(v_core_mask);
        v_core_mask &= ~(static_cast<u64>(1) << next);
        p_core_mask |=
            (static_cast<u64>(1) << VirtualToPhysicalCore(profile, static_cast<s32>(next)));
    }
    return p_core_mask;
}

// Cortex-A57 supports 4 memory watchpoints
constexpr u64 NUM_WATCHPOINTS = 4;

} // namespace Hardware

} // namespace Core
