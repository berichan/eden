// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <span>
#include "common/common_types.h"

namespace Service::GSV {

struct NativePresentState {
    std::array<u16, 4> crop;
    u64 flags;
    u64 submitted_ticks;
    u64 swap_interval;
    u64 hdr_flags;
    std::array<u16, 4> hdr_luminance;
    u64 recorder_flags;
    u64 user_data;

    bool operator==(const NativePresentState&) const = default;
};
static_assert(sizeof(NativePresentState) == 64);

struct NativePresentSync {
    u32 counter;
    u8 mode;
    std::array<u8, 3> padding;
    u64 target;

    // The SDK's empty handle is sixteen zero bytes. An invalid counter or
    // unsupported mode must not become an empty fence by normalization.
    bool IsEmpty() const {
        return counter == 0 && mode == 0 && padding == std::array<u8, 3>{} && target == 0;
    }
};
static_assert(sizeof(NativePresentSync) == 16);

struct NativePresentPacket {
    u64 layer_token;
    u8 slot;
    u32 image;
    NativePresentState state;
    NativePresentSync render_sync;
};

struct NativeCancelPacket {
    u64 layer_token;
    u8 slot;
    NativePresentState state;
};

namespace PresentPacketDetail {
// Call only after validating the complete fixed-size payload. The input is
// little-endian wire data, independent of host structure padding and alignment.
template <typename T>
inline T Read(std::span<const u8> data, size_t offset) {
    T value{};
    for (size_t byte = 0; byte < sizeof(T); ++byte)
        value = static_cast<T>(value | (T{data[offset + byte]} << (byte * 8)));
    return value;
}

inline NativePresentState ReadState(std::span<const u8> data, size_t base) {
    NativePresentState state{};
    for (size_t index = 0; index < 4; ++index) {
        state.crop[index] = Read<u16>(data, base + index * 2);
        state.hdr_luminance[index] = Read<u16>(data, base + 40 + index * 2);
    }
    state.flags = Read<u64>(data, base + 8);
    state.submitted_ticks = Read<u64>(data, base + 16);
    state.swap_interval = Read<u64>(data, base + 24);
    state.hdr_flags = Read<u64>(data, base + 32);
    state.recorder_flags = Read<u64>(data, base + 48);
    state.user_data = Read<u64>(data, base + 56);
    return state;
}
} // namespace PresentPacketDetail

// Decode the payload after the eight-byte plain TIPC header. This does not
// establish layer/image ownership, slot availability, supported presentation
// state, valid syncpoint generations, backing or rendering completion.
inline std::optional<NativePresentPacket> DecodeNativePresentPacket(std::span<const u8> data) {
    if (data.size() != 96)
        return std::nullopt;
    using PresentPacketDetail::Read;
    return NativePresentPacket{
        .layer_token = Read<u64>(data, 0),
        // The sender initializes one byte of this four-byte slot.
        .slot = data[8],
        .image = Read<u32>(data, 12),
        .state = PresentPacketDetail::ReadState(data, 16),
        .render_sync = {Read<u32>(data, 80),
                        data[84],
                        {data[85], data[86], data[87]},
                        Read<u64>(data, 88)},
    };
}

inline std::optional<NativeCancelPacket> DecodeNativeCancelPacket(std::span<const u8> data) {
    if (data.size() != 76)
        return std::nullopt;
    using PresentPacketDetail::Read;
    // Cancel has neither an image handle nor an appended rendering fence.
    return NativeCancelPacket{
        .layer_token = Read<u64>(data, 0),
        .slot = data[8],
        .state = PresentPacketDetail::ReadState(data, 12),
    };
}

} // namespace Service::GSV
