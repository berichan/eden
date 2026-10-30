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

TEST_CASE("Native SWDX bundle initialization retains each independently masked context field",
          "[ipc][video_core]") {
    Config config;
    Host host;
    constexpr std::array<std::array<u32, 3>, 22> writes{{
        {0x418e40, 0x7, 0xf},
        {0x418e58, 0x842, 0xffff},
        {0x418e40, 0x70, 0xf0},
        {0x418e58, 0x04f10000, 0xffff0000},
        {0x418e40, 0x700, 0xf00},
        {0x418e5c, 0x53, 0xffff},
        {0x418e40, 0x7000, 0xf000},
        {0x418e5c, 0x00e90000, 0xffff0000},
        {0x418e40, 0x70000, 0xf0000},
        {0x418e60, 0xea, 0xffff},
        {0x418e40, 0x700000, 0xf00000},
        {0x418e60, 0x00eb0000, 0xffff0000},
        {0x418e40, 0x7000000, 0xf000000},
        {0x418e64, 0x208, 0xffff},
        {0x418e40, 0x70000000, 0xf0000000},
        {0x418e64, 0x02090000, 0xffff0000},
        {0x418e44, 0x7, 0xf},
        {0x418e68, 0x20a, 0xffff},
        {0x418e44, 0x70, 0xf0},
        {0x418e68, 0x020b0000, 0xffff0000},
        {0x418e44, 0x700, 0xf00},
        {0x418e6c, 0x644, 0xffff},
    }};
    REQUIRE(config.ApplyMaskedRegister(0x418800, 1, 1, host.Bind()) == Config::Error::None);
    REQUIRE(config.SwdxBundle(0x418e40)->known_mask == 0);
    for (const auto& write : writes) {
        REQUIRE(config.ApplyMaskedRegister(write[0], write[1], write[2], host.Bind()) ==
                Config::Error::None);
    }
    REQUIRE(config.SwdxBundle(0x418e40)->value == 0x77777777);
    REQUIRE(config.SwdxBundle(0x418e40)->known_mask == 0xffffffff);
    REQUIRE(config.SwdxBundle(0x418e44)->value == 0x777);
    REQUIRE(config.SwdxBundle(0x418e44)->known_mask == 0xfff);
    REQUIRE(config.SwdxBundle(0x418e58)->value == 0x04f10842);
    REQUIRE(config.SwdxBundle(0x418e5c)->value == 0x00e90053);
    REQUIRE(config.SwdxBundle(0x418e60)->value == 0x00eb00ea);
    REQUIRE(config.SwdxBundle(0x418e64)->value == 0x02090208);
    REQUIRE(config.SwdxBundle(0x418e68)->value == 0x020b020a);
    REQUIRE(config.SwdxBundle(0x418e6c)->value == 0x644);
    REQUIRE(config.SwdxBundle(0x418e6c)->known_mask == 0xffff);
    REQUIRE(config.PolygonOffsetNanIsZero());
    REQUIRE(host.waits == 23);
}

TEST_CASE("Native SWDX bundle writes cannot acknowledge unsupported or unfinished settings",
          "[ipc][video_core]") {
    Config config;
    Host host;
    for (u32 address : {0x418e48U, 0x418e4cU, 0x418e50U, 0x418e54U, 0x418e70U}) {
        REQUIRE_FALSE(config.SwdxBundle(address));
        REQUIRE(config.ApplyMaskedRegister(address, 7, 0xf, host.Bind()) ==
                Config::Error::UnsupportedRegister);
    }
    REQUIRE(config.ApplyMaskedRegister(0x418e44, 0x7000, 0xf000, host.Bind()) ==
            Config::Error::UnsupportedMask);
    REQUIRE(config.ApplyMaskedRegister(0x418e40, 0x77, 0xff, host.Bind()) ==
            Config::Error::UnsupportedMask);
    REQUIRE(config.ApplyMaskedRegister(0x418e6c, 0, 0xffff0000, host.Bind()) ==
            Config::Error::UnsupportedMask);
    REQUIRE(config.ApplyMaskedRegister(0x418e40, 0xf, 0xf, host.Bind()) ==
            Config::Error::UnsupportedValue);
    REQUIRE(host.waits == 0);
    host.accept = false;
    REQUIRE(config.ApplyMaskedRegister(0x418e40, 7, 0xf, host.Bind()) == Config::Error::WaitFailed);
    REQUIRE(config.SwdxBundle(0x418e40)->known_mask == 0);
    host.accept = true;
    host.cancel_during_wait = true;
    REQUIRE(config.ApplyMaskedRegister(0x418e40, 7, 0xf, host.Bind()) == Config::Error::Cancelled);
    REQUIRE(config.SwdxBundle(0x418e40)->known_mask == 0);
    REQUIRE(config.SwdxBundle(0x418e40)->value == 0);
}
