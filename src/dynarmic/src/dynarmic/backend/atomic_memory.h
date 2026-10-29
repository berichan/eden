// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/assert.h"
#include "dynarmic/interface/A64/config.h"
#include "dynarmic/interface/exclusive_monitor.h"

namespace Dynarmic::Backend {

template<auto callback, typename T>
T CompareExchangeMemory(const A64::UserConfig* conf, A64::VAddr address, T expected, T value) {
    const auto operation = [&] { return (conf->callbacks->*callback)(address, expected, value); };
    return conf->global_monitor ? conf->global_monitor->DoAtomicOperation(operation) : operation();
}

template<auto callback, typename T>
T AtomicMemory(const A64::UserConfig* conf, A64::VAddr address, T operand, u8 operation) {
    const auto perform = [&] {
        // An unchanged zero exchange supplies an atomic initial observation.
        T observed = (conf->callbacks->*callback)(address, T{}, T{});
        for (;;) {
            T value{};
            constexpr T sign = T{1} << (sizeof(T) * 8 - 1);
            switch (operation) {
            case 0:
                value = static_cast<T>(observed + operand);
                break;
            case 1:
                value = static_cast<T>(observed & ~operand);
                break;
            case 2:
                value = static_cast<T>(observed ^ operand);
                break;
            case 3:
                value = static_cast<T>(observed | operand);
                break;
            case 4:
                value = (observed ^ sign) > (operand ^ sign) ? observed : operand;
                break;
            case 5:
                value = (observed ^ sign) < (operand ^ sign) ? observed : operand;
                break;
            case 6:
                value = observed > operand ? observed : operand;
                break;
            case 7:
                value = observed < operand ? observed : operand;
                break;
            case 8:
                value = operand;
                break;
            default:
                UNREACHABLE();
            }
            const T previous = (conf->callbacks->*callback)(address, observed, value);
            if (previous == observed) {
                return observed;
            }
            observed = previous;
        }
    };
    return conf->global_monitor ? conf->global_monitor->DoAtomicOperation(perform) : perform();
}

}  // namespace Dynarmic::Backend
