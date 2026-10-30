// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "core/hle/service/audio/audio_in_manager.h"
#include "core/hle/service/audio/audio_out_manager.h"
#include "core/hle/service/audio/audio_renderer_manager.h"
#include "core/hle/service/audio/audio_user_session.h"
#include "core/hle/service/cmif_serialization.h"

namespace Service::Audio {

IAudioUserSession::IAudioUserSession(Core::System& system_, std::shared_ptr<IAudioInManager> input_,
                                     std::shared_ptr<IAudioOutManager> output_,
                                     std::shared_ptr<IAudioRendererManager> renderer_)
    : ServiceFramework{system_, "aud:u"}, input{std::move(input_)}, output{std::move(output_)},
      renderer{std::move(renderer_)} {
    // IUserClientSession uses CMIF commands with no input and one returned interface.
    static const FunctionInfo functions[] = {
        {0, D<&IAudioUserSession::GetAudioInManager>, "GetAudioInManager"},
        {1, D<&IAudioUserSession::GetAudioOutManager>, "GetAudioOutManager"},
        {2, D<&IAudioUserSession::GetAudioRendererManager>, "GetAudioRendererManager"},
    };
    RegisterHandlers(functions);
}

IAudioUserSession::~IAudioUserSession() = default;

Result IAudioUserSession::GetAudioInManager(Out<SharedPointer<IAudioInManager>> out_manager) {
    LOG_DEBUG(Service_Audio, "Opening shared audio input manager");
    *out_manager = input;
    R_SUCCEED();
}

Result IAudioUserSession::GetAudioOutManager(Out<SharedPointer<IAudioOutManager>> out_manager) {
    LOG_DEBUG(Service_Audio, "Opening shared audio output manager");
    *out_manager = output;
    R_SUCCEED();
}

Result IAudioUserSession::GetAudioRendererManager(
    Out<SharedPointer<IAudioRendererManager>> out_manager) {
    LOG_DEBUG(Service_Audio, "Opening shared audio renderer manager");
    *out_manager = renderer;
    R_SUCCEED();
}

} // namespace Service::Audio
