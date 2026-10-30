// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include "video_core/control/native_atomic_sequence.h"

TEST_CASE("Native atomic scopes preserve ownership until a matching end", "[ipc][video_core]") {
    Tegra::Control::NativeAtomicSequence sequence;
    REQUIRE_FALSE(sequence.IsActive());
    REQUIRE_FALSE(sequence.End(0, 1));
    REQUIRE_FALSE(sequence.Begin(8, 1));
    REQUIRE_FALSE(sequence.Begin(0, 2));
    for (u32 value : {0U, 1U}) {
        REQUIRE(sequence.Begin(3, value));
        REQUIRE(sequence.IsActive());
        REQUIRE(sequence.AllowsSubchannel(3));
        REQUIRE_FALSE(sequence.AllowsSubchannel(0));
        REQUIRE_FALSE(sequence.Begin(3, value));
        REQUIRE_FALSE(sequence.Begin(0, value));
        REQUIRE_FALSE(sequence.End(0, value));
        REQUIRE_FALSE(sequence.End(3, value ^ 1));
        REQUIRE(sequence.IsActive());
        REQUIRE(sequence.End(3, value));
        REQUIRE_FALSE(sequence.IsActive());
        REQUIRE(sequence.AllowsSubchannel(0));
    }
}
