// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include "video_core/control/native_graphics_config.h"

namespace {
using Config = Tegra::Control::NativeGraphicsConfig;
struct Host {
    u32 waits{};
    bool accept{true};
    bool cancelled{};
    bool cancel_during_wait{};
    Config::Host Bind() {
        return {this,
                [](void* context) {
                    auto& self = *static_cast<Host*>(context);
                    ++self.waits;
                    self.cancelled = self.cancel_during_wait;
                    return self.accept;
                },
                [](void* context) { return static_cast<Host*>(context)->cancelled; }};
    }
};
} // namespace

TEST_CASE("Native setup register updates apply modeled context state only after idle",
          "[ipc][video_core]") {
    Config config;
    Host host;
    REQUIRE_FALSE(config.PolygonOffsetNanIsZero());
    REQUIRE(config.ApplyMaskedRegister(0x418800, 1, 1, host.Bind()) == Config::Error::None);
    REQUIRE(config.PolygonOffsetNanIsZero());
    REQUIRE(host.waits == 1);
    REQUIRE(config.ApplyMaskedRegister(0x418800, 0, 1, host.Bind()) == Config::Error::None);
    REQUIRE_FALSE(config.PolygonOffsetNanIsZero());
    REQUIRE(host.waits == 2);
    host.accept = false;
    REQUIRE(config.ApplyMaskedRegister(0x418800, 1, 1, host.Bind()) == Config::Error::WaitFailed);
    REQUIRE_FALSE(config.PolygonOffsetNanIsZero());
    REQUIRE(host.waits == 3);
    REQUIRE(config.ApplyMaskedRegister(0x418800, 1, 1, {}) == Config::Error::WaitFailed);
    REQUIRE_FALSE(config.PolygonOffsetNanIsZero());
}

TEST_CASE("Unsupported or cancelled native setup updates preserve existing context state",
          "[ipc][video_core]") {
    Config config;
    Host host;
    REQUIRE(config.ApplyMaskedRegister(0x418800, 1, 1, host.Bind()) == Config::Error::None);
    REQUIRE(config.ApplyMaskedRegister(0x419f78, 0, 8, host.Bind()) ==
            Config::Error::UnsupportedRegister);
    REQUIRE(config.ApplyMaskedRegister(0x418800, 1, 3, host.Bind()) ==
            Config::Error::UnsupportedMask);
    REQUIRE(config.ApplyMaskedRegister(0x418800, 2, 1, host.Bind()) ==
            Config::Error::UnsupportedValue);
    REQUIRE(config.PolygonOffsetNanIsZero());
    REQUIRE(host.waits == 1);
    host.cancelled = true;
    REQUIRE(config.ApplyMaskedRegister(0x418800, 0, 1, host.Bind()) == Config::Error::Cancelled);
    REQUIRE(host.waits == 1);
    host.cancelled = false;
    host.cancel_during_wait = true;
    REQUIRE(config.ApplyMaskedRegister(0x418800, 0, 1, host.Bind()) == Config::Error::Cancelled);
    REQUIRE(host.waits == 2);
    REQUIRE(config.PolygonOffsetNanIsZero());
}

TEST_CASE("Native conservative setup preserves the polygon offset field in the same register",
          "[ipc][video_core]") {
    Config config;
    Host host;
    REQUIRE(config.ApplyMaskedRegister(0x418800, 1, 1, host.Bind()) == Config::Error::None);
    host.accept = false;
    REQUIRE(config.ApplyMaskedRegister(0x418800, 0, 0x1800000, host.Bind()) ==
            Config::Error::WaitFailed);
    REQUIRE(config.SetupDebugKnownMask() == 1);
    host.accept = true;
    REQUIRE(config.ApplyMaskedRegister(0x418800, 0, 0x1800000, host.Bind()) == Config::Error::None);
    REQUIRE(config.PolygonOffsetNanIsZero());
    REQUIRE(config.SetupDebug() == 1);
    REQUIRE(config.SetupDebugKnownMask() == 0x1800001);
    REQUIRE(config.ApplyMaskedRegister(0x418800, 0x800000, 0x1800000, host.Bind()) ==
            Config::Error::UnsupportedValue);
    REQUIRE(config.ApplyMaskedRegister(0x418800, 0, 0x1800001, host.Bind()) ==
            Config::Error::UnsupportedMask);
    REQUIRE(host.waits == 3);
    REQUIRE(config.ApplyMaskedRegister(0x418800, 0, 1, host.Bind()) == Config::Error::None);
    REQUIRE_FALSE(config.PolygonOffsetNanIsZero());
    REQUIRE(config.SetupDebugKnownMask() == 0x1800001);
}

TEST_CASE("Native texture setup records only established initialization bits",
          "[ipc][video_core]") {
    Config config;
    Host host;
    REQUIRE(config.TextureSamplerDebugKnownMask() == 0);
    host.accept = false;
    REQUIRE(config.ApplyMaskedRegister(0x419a08, 0, 0x10, host.Bind()) ==
            Config::Error::WaitFailed);
    REQUIRE(config.TextureSamplerDebugKnownMask() == 0);
    host.accept = true;
    REQUIRE(config.ApplyMaskedRegister(0x419a08, 0, 0x10, host.Bind()) == Config::Error::None);
    REQUIRE(config.TextureSamplerDebug() == 0);
    REQUIRE(config.TextureSamplerDebugKnownMask() == 0x10);
    REQUIRE(config.ApplyMaskedRegister(0x419a08, 0x10, 0x10, host.Bind()) ==
            Config::Error::UnsupportedValue);
    REQUIRE(config.ApplyMaskedRegister(0x419a08, 0, 0x11, host.Bind()) ==
            Config::Error::UnsupportedMask);
    REQUIRE(config.TextureSamplerDebugKnownMask() == 0x10);
    REQUIRE(host.waits == 2);
}

TEST_CASE("Native helper loads require the Ampere register and successful idle",
          "[ipc][video_core]") {
    Config config;
    Host host;
    REQUIRE_FALSE(config.HelperInvocationLoadsEnabled());
    REQUIRE(config.ApplyMaskedRegister(0x419f78, 0, 8, host.Bind()) ==
            Config::Error::UnsupportedRegister);
    REQUIRE(config.ApplyMaskedRegister(0x419ba4, 0, 9, host.Bind()) ==
            Config::Error::UnsupportedMask);
    REQUIRE(config.ApplyMaskedRegister(0x419ba4, 8, 8, host.Bind()) ==
            Config::Error::UnsupportedValue);
    REQUIRE(host.waits == 0);
    host.accept = false;
    REQUIRE(config.ApplyMaskedRegister(0x419ba4, 0, 8, host.Bind()) == Config::Error::WaitFailed);
    REQUIRE_FALSE(config.HelperInvocationLoadsEnabled());
    host.accept = true;
    host.cancel_during_wait = true;
    REQUIRE(config.ApplyMaskedRegister(0x419ba4, 0, 8, host.Bind()) == Config::Error::Cancelled);
    REQUIRE_FALSE(config.HelperInvocationLoadsEnabled());
    host.cancelled = false;
    host.cancel_during_wait = false;
    REQUIRE(config.ApplyMaskedRegister(0x419ba4, 0, 8, host.Bind()) == Config::Error::None);
    REQUIRE(config.HelperInvocationLoadsEnabled());
    REQUIRE(host.waits == 3);
}

TEST_CASE("Native texture LOD initialization preserves separate masked fields",
          "[ipc][video_core]") {
    Config config;
    Host host;
    REQUIRE(config.TextureLodDebugKnownMask() == 0);
    REQUIRE(config.ApplyMaskedRegister(0x419a04, 1, 1, host.Bind()) == Config::Error::None);
    REQUIRE(config.TextureLodDebug() == 1);
    REQUIRE(config.TextureLodDebugKnownMask() == 1);
    host.accept = false;
    REQUIRE(config.ApplyMaskedRegister(0x419a04, 2, 2, host.Bind()) == Config::Error::WaitFailed);
    REQUIRE(config.TextureLodDebug() == 1);
    REQUIRE(config.TextureLodDebugKnownMask() == 1);
    host.accept = true;
    REQUIRE(config.ApplyMaskedRegister(0x419a04, 2, 2, host.Bind()) == Config::Error::None);
    REQUIRE(config.TextureLodDebug() == 3);
    REQUIRE(config.TextureLodDebugKnownMask() == 3);
    REQUIRE(config.ApplyMaskedRegister(0x419a04, 3, 3, host.Bind()) ==
            Config::Error::UnsupportedMask);
    REQUIRE(config.ApplyMaskedRegister(0x419a04, 0, 1, host.Bind()) ==
            Config::Error::UnsupportedValue);
    REQUIRE(config.ApplyMaskedRegister(0x419a04, 3, 2, host.Bind()) ==
            Config::Error::UnsupportedValue);
    REQUIRE(host.waits == 3);
    REQUIRE(config.TextureLodDebug() == 3);
}
