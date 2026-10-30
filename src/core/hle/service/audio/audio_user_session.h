// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <memory>
#include "core/hle/service/cmif_types.h"
#include "core/hle/service/service.h"

namespace Service::Audio {

class IAudioInManager;
class IAudioOutManager;
class IAudioRendererManager;

class IAudioUserSession final : public ServiceFramework<IAudioUserSession> {
public:
    IAudioUserSession(Core::System& system, std::shared_ptr<IAudioInManager> input,
                      std::shared_ptr<IAudioOutManager> output,
                      std::shared_ptr<IAudioRendererManager> renderer);
    ~IAudioUserSession() override;

private:
    Result GetAudioInManager(Out<SharedPointer<IAudioInManager>> out_manager);
    Result GetAudioOutManager(Out<SharedPointer<IAudioOutManager>> out_manager);
    Result GetAudioRendererManager(Out<SharedPointer<IAudioRendererManager>> out_manager);

    const std::shared_ptr<IAudioInManager> input;
    const std::shared_ptr<IAudioOutManager> output;
    const std::shared_ptr<IAudioRendererManager> renderer;
};

} // namespace Service::Audio
