// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include "video_core/control/native_cache.h"

TEST_CASE("Native cache actions decode only supported C797 invalidation fields",
          "[ipc][video_core]") {
    using Tegra::Control::DecodeNativeGraphicsCacheAction;
    const auto shader = DecodeNativeGraphicsCacheAction(0x369, 0x1011);
    REQUIRE(shader.has_value());
    REQUIRE(shader->instructions);
    REQUIRE(shader->global_data);
    REQUIRE(shader->constants);
    REQUIRE_FALSE(shader->texture_data);
    REQUIRE_FALSE(shader->texture_headers);
    REQUIRE_FALSE(shader->samplers);
    const auto instruction = DecodeNativeGraphicsCacheAction(0x369, 1);
    REQUIRE(instruction->instructions);
    REQUIRE_FALSE(instruction->global_data);
    REQUIRE_FALSE(instruction->constants);
    REQUIRE(DecodeNativeGraphicsCacheAction(0x4a2, 0)->texture_data);
    REQUIRE(DecodeNativeGraphicsCacheAction(0x50a, 0)->texture_headers);
    REQUIRE(DecodeNativeGraphicsCacheAction(0x509, 0)->samplers);
    for (u32 method : {0x4a2U, 0x50aU, 0x509U}) {
        REQUIRE_FALSE(DecodeNativeGraphicsCacheAction(method, 1));
        REQUIRE_FALSE(DecodeNativeGraphicsCacheAction(method, 0x10));
    }
    REQUIRE_FALSE(DecodeNativeGraphicsCacheAction(0x369, 2));
    REQUIRE_FALSE(DecodeNativeGraphicsCacheAction(0x369, 4));
    REQUIRE_FALSE(DecodeNativeGraphicsCacheAction(0x369, 0x10000));
    REQUIRE_FALSE(DecodeNativeGraphicsCacheAction(0x87, 0x1011));
    REQUIRE_FALSE(DecodeNativeGraphicsCacheAction(0x674, 0));
}
