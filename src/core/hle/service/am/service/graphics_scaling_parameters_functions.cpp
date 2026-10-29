// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "core/hle/service/am/applet.h"
#include "core/hle/service/am/service/graphics_scaling_parameters_functions.h"
#include "core/hle/service/cmif_serialization.h"

namespace Service::AM {

IGraphicsScalingParametersFunctions::IGraphicsScalingParametersFunctions(
    Core::System& system_, std::shared_ptr<Applet> applet)
    : ServiceFramework{system_, "IGraphicsScalingParametersFunctions"},
      m_applet{std::move(applet)} {
    static const FunctionInfo functions[] = {
        {10, D<&IGraphicsScalingParametersFunctions::GetGraphicsScalingParameters>,
         "GetGraphicsScalingParameters"},
        {1050, D<&IGraphicsScalingParametersFunctions::SetGraphicsScaleChangedNotificationEnabled>,
         "SetGraphicsScaleChangedNotificationEnabled"},
    };
    RegisterHandlers(functions);
}

IGraphicsScalingParametersFunctions::~IGraphicsScalingParametersFunctions() = default;

Result IGraphicsScalingParametersFunctions::GetGraphicsScalingParameters(
    Out<GraphicsScalingParameters> out_parameters) {
    // Eden does not share the foreground GPU budget with other applets. Selector
    // 10 is the default quota and the full (1:1) screen-ratio entry.
    *out_parameters = {.gpu_quota = 10, .unknown = 0, .screen_ratio = 10};
    LOG_DEBUG(Service_AM, "called, gpu_quota={}, screen_ratio={}", out_parameters->gpu_quota,
              out_parameters->screen_ratio);
    R_SUCCEED();
}

Result IGraphicsScalingParametersFunctions::SetGraphicsScaleChangedNotificationEnabled(
    bool enabled) {
    LOG_DEBUG(Service_AM, "called, enabled={}", enabled);
    std::scoped_lock lk{m_applet->lock};
    m_applet->graphics_scale_changed_notification_enabled = enabled;
    R_SUCCEED();
}

} // namespace Service::AM
