// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "core/hle/service/cmif_types.h"
#include "core/hle/service/service.h"

namespace Service::AM {

struct Applet;

struct GraphicsScalingParameters {
    u32 gpu_quota;
    u32 unknown;
    u32 screen_ratio;
};
static_assert(sizeof(GraphicsScalingParameters) == 0xc);

class IGraphicsScalingParametersFunctions final
    : public ServiceFramework<IGraphicsScalingParametersFunctions> {
public:
    explicit IGraphicsScalingParametersFunctions(Core::System& system,
                                                 std::shared_ptr<Applet> applet);
    ~IGraphicsScalingParametersFunctions() override;

private:
    Result GetGraphicsScalingParameters(Out<GraphicsScalingParameters> out_parameters);
    Result SetGraphicsScaleChangedNotificationEnabled(bool enabled);

    const std::shared_ptr<Applet> m_applet;
};

} // namespace Service::AM
