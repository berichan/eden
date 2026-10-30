// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstring>
#include <catch2/catch_test_macros.hpp>
#include "audio_core/common/feature_support.h"
#include "audio_core/renderer/behavior/behavior_info.h"
#include "audio_core/renderer/behavior/info_updater.h"

namespace {

constexpr u32 Revision(char family, u32 version) {
    return Common::MakeMagic('R', 'E', family, static_cast<char>('0' + version));
}

using AudioCore::Renderer::BehaviorInfo;

} // namespace

TEST_CASE("Audio wire revisions retain their family and numeric feature version", "[audio][ipc]") {
    for (const u32 version : {1U, 8U, AudioCore::CurrentRevision}) {
        BehaviorInfo legacy;
        legacy.SetUserLibRevision(Revision('V', version));
        for (const char family : {'V', 'v'}) {
            const u32 revision = Revision(family, version);
            REQUIRE(AudioCore::CheckValidWireRevision(revision));
            BehaviorInfo behavior;
            behavior.SetUserLibRevision(revision);
            REQUIRE(behavior.GetUserRevision() == revision);
            REQUIRE(behavior.GetUserRevisionNum() == version);
            REQUIRE(behavior.GetProcessRevisionNum() == AudioCore::CurrentRevision);
            REQUIRE(behavior.GetProcessRevision() == Revision(family, AudioCore::CurrentRevision));
            REQUIRE(behavior.IsWaveBufferVer2Supported() == legacy.IsWaveBufferVer2Supported());
            REQUIRE(behavior.IsVoiceInParameterV2Supported() ==
                    legacy.IsVoiceInParameterV2Supported());
        }
        BehaviorInfo numeric;
        numeric.SetUserLibRevision(version);
        REQUIRE(numeric.GetUserRevision() == Revision('V', version));
        numeric.SetUserLibRevision(Revision('v', version));
        REQUIRE(numeric.GetUserRevision() == Revision('v', version));
        REQUIRE(numeric.GetProcessRevision() == Revision('v', AudioCore::CurrentRevision));
        numeric.SetUserLibRevision(Revision('V', version));
        REQUIRE(numeric.GetProcessRevision() == Revision('V', AudioCore::CurrentRevision));
    }
    for (const u32 invalid : {0U, 1U, Revision('V', 0), Revision('v', 0), Revision('x', 1),
                              Revision('V', AudioCore::CurrentRevision + 1),
                              Revision('v', AudioCore::CurrentRevision + 1)}) {
        REQUIRE_FALSE(AudioCore::CheckValidWireRevision(invalid));
    }
}

TEST_CASE("Truncated audio updates cannot write a header or change behavior state",
          "[audio][ipc]") {
    for (const size_t input_size :
         {size_t{0}, size_t{0x3f}, size_t{0x40}, size_t{0x4f}, size_t{0x50}}) {
        for (const size_t output_size :
             {size_t{0}, size_t{0x3f}, size_t{0x40}, size_t{0xef}, size_t{0xf0}}) {
            if (input_size == 0x50 && output_size == 0xf0) {
                continue;
            }
            CAPTURE(input_size, output_size);
            BehaviorInfo behavior;
            behavior.SetUserLibRevision(Revision('v', 8));
            alignas(16) std::array<u8, 0x50> input{};
            alignas(16) std::array<u8, 0x100> output;
            output.fill(0x5a);
            const auto unchanged = output;
            AudioCore::Renderer::InfoUpdater updater{std::span{input}.first(input_size),
                                                     std::span{output}.first(output_size), nullptr,
                                                     behavior};
            REQUIRE(updater.UpdateBehaviorInfo(behavior) ==
                    Service::Audio::ResultInvalidUpdateInfo);
            REQUIRE(output == unchanged);
            REQUIRE_FALSE(behavior.IsMemoryForceMappingEnabled());
        }
    }
}

TEST_CASE("Audio behavior updates require the negotiated wire revision before mutating state",
          "[audio][ipc]") {
    for (const char family : {'V', 'v'}) {
        const u32 revision = Revision(family, 8);
        for (const u32 incoming :
             {revision, Revision(family == 'V' ? 'v' : 'V', 8), Revision(family, 9),
              Revision('x', 8), Revision(family, AudioCore::CurrentRevision + 1)}) {
            for (const u32 size : {0x10U, 0x18U}) {
                CAPTURE(family, incoming, size);
                BehaviorInfo behavior;
                behavior.SetUserLibRevision(revision);
                behavior.AppendError({Service::Audio::ResultInvalidArgument, 0, 0x1000});
                alignas(16) std::array<u8, 0x50> input{};
                alignas(16) std::array<u8, 0x100> output{};
                std::memcpy(input.data(), &incoming, sizeof(incoming));
                std::memcpy(input.data() + 4, &size, sizeof(size));
                BehaviorInfo::InParameter parameter{};
                parameter.revision = incoming;
                parameter.flags.IsMemoryForceMappingEnabled = true;
                std::memcpy(input.data() + 0x40, &parameter, sizeof(parameter));
                AudioCore::Renderer::InfoUpdater updater{input, output, nullptr, behavior};
                u32 output_revision{};
                std::memcpy(&output_revision, output.data(), sizeof(output_revision));
                REQUIRE(output_revision == Revision(family, AudioCore::CurrentRevision));
                const auto result = updater.UpdateBehaviorInfo(behavior);
                const bool valid = incoming == revision && size == sizeof(parameter);
                REQUIRE(result ==
                        (valid ? ResultSuccess : Service::Audio::ResultInvalidUpdateInfo));
                REQUIRE(behavior.IsMemoryForceMappingEnabled() == valid);
                std::array<BehaviorInfo::ErrorInfo, 10> errors{};
                u32 error_count{};
                behavior.CopyErrorInfo(errors, error_count);
                REQUIRE(error_count == (valid ? 0U : 1U));
                if (!valid) {
                    REQUIRE(errors[0].error_code == Service::Audio::ResultInvalidArgument);
                    REQUIRE(errors[0].address == 0x1000);
                }
            }
        }
    }
}
