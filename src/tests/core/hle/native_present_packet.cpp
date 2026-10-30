// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include "core/hle/service/gsv/native_present_packet.h"

using namespace Service::GSV;

TEST_CASE("Native present packets retain state, byte slots and full sync targets", "[ipc][gsv]") {
    std::array<u8, 96> data{};
    for (u32 index = 0; index < data.size(); ++index)
        data[index] = static_cast<u8>(index);
    const auto packet = DecodeNativePresentPacket(data);
    REQUIRE(packet);
    REQUIRE(packet->layer_token == 0x0706050403020100ULL);
    REQUIRE(packet->slot == 8);
    REQUIRE(packet->image == 0x0f0e0d0c);
    REQUIRE(packet->state.crop == std::array<u16, 4>{0x1110, 0x1312, 0x1514, 0x1716});
    REQUIRE(packet->state.flags == 0x1f1e1d1c1b1a1918ULL);
    REQUIRE(packet->state.submitted_ticks == 0x2726252423222120ULL);
    REQUIRE(packet->state.swap_interval == 0x2f2e2d2c2b2a2928ULL);
    REQUIRE(packet->state.hdr_flags == 0x3736353433323130ULL);
    REQUIRE(packet->state.hdr_luminance == std::array<u16, 4>{0x3938, 0x3b3a, 0x3d3c, 0x3f3e});
    REQUIRE(packet->state.recorder_flags == 0x4746454443424140ULL);
    REQUIRE(packet->state.user_data == 0x4f4e4d4c4b4a4948ULL);
    REQUIRE(packet->render_sync.counter == 0x53525150);
    REQUIRE(packet->render_sync.mode == 84);
    REQUIRE(packet->render_sync.padding == std::array<u8, 3>{85, 86, 87});
    REQUIRE(packet->render_sync.target == 0x5f5e5d5c5b5a5958ULL);
    REQUIRE_FALSE(packet->render_sync.IsEmpty());
    data[9] = data[10] = data[11] = 0xff;
    const auto padded = DecodeNativePresentPacket(data);
    REQUIRE(padded);
    REQUIRE(padded->slot == packet->slot);
    REQUIRE(padded->state == packet->state);
}

TEST_CASE("Native cancel packets put state immediately after the byte slot", "[ipc][gsv]") {
    std::array<u8, 96> present{};
    for (u32 index = 0; index < present.size(); ++index)
        present[index] = static_cast<u8>(index);
    std::array<u8, 76> cancel{};
    std::copy_n(present.begin(), 12, cancel.begin());
    std::copy_n(present.begin() + 16, 64, cancel.begin() + 12);
    const auto queued = DecodeNativePresentPacket(present);
    const auto cancelled = DecodeNativeCancelPacket(cancel);
    REQUIRE(queued);
    REQUIRE(cancelled);
    REQUIRE(cancelled->layer_token == queued->layer_token);
    REQUIRE(cancelled->slot == queued->slot);
    REQUIRE(cancelled->state == queued->state);
    REQUIRE_FALSE(DecodeNativePresentPacket(cancel));
    REQUIRE_FALSE(DecodeNativeCancelPacket(present));
}

TEST_CASE("Native present packet decoders require exact payload sizes", "[ipc][gsv]") {
    std::array<u8, 97> data{};
    for (size_t size = 0; size < data.size(); ++size) {
        REQUIRE(DecodeNativePresentPacket(std::span{data}.first(size)).has_value() == (size == 96));
        REQUIRE(DecodeNativeCancelPacket(std::span{data}.first(size)).has_value() == (size == 76));
    }
    REQUIRE_FALSE(DecodeNativePresentPacket(data));
    REQUIRE_FALSE(DecodeNativeCancelPacket(data));
    const auto packet = DecodeNativePresentPacket(std::span{data}.first(96));
    REQUIRE(packet);
    REQUIRE(packet->render_sync.IsEmpty());
    data[80] = 0xff;
    REQUIRE_FALSE(DecodeNativePresentPacket(std::span{data}.first(96))->render_sync.IsEmpty());
    data[80] = 0;
    data[84] = 1;
    REQUIRE_FALSE(DecodeNativePresentPacket(std::span{data}.first(96))->render_sync.IsEmpty());
    data[84] = 0;
    data[85] = 1;
    REQUIRE_FALSE(DecodeNativePresentPacket(std::span{data}.first(96))->render_sync.IsEmpty());
}
