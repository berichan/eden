// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "core/hle/service/am/service/performance_functions.h"
#include "core/hle/service/cmif_serialization.h"

namespace Service::AM {

IPerformanceFunctions::IPerformanceFunctions(Core::System& system_)
    : ServiceFramework{system_, "IPerformanceFunctions"} {
    static const FunctionInfo functions[] = {
        {0, D<&IPerformanceFunctions::IsCpuOverclockEnabled>, "IsCpuOverclockEnabled"},
        {1, nullptr, "SetCpuOverclockEnabled"},
    };
    RegisterHandlers(functions);
}

IPerformanceFunctions::~IPerformanceFunctions() = default;

Result IPerformanceFunctions::IsCpuOverclockEnabled(Out<bool> out_enabled) {
    // The current guest clock model does not enable CPU overclocking.
    *out_enabled = false;
    LOG_DEBUG(Service_AM, "called, enabled={}", *out_enabled);
    R_SUCCEED();
}

} // namespace Service::AM
