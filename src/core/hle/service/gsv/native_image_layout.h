// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include "common/common_types.h"
#include "common/swap.h"
#include "core/hle/result.h"

namespace Service::GSV {

struct NativeImagePreference {
    u32_le reserved;
    u32_le type;
    u32_le packed_extent;
    u32_le reserved_0c;
    u32_le row_alignment_log2;
    u32_le pitch_gobs;
    u32_le format;
    u32_le options;
    std::array<u64_le, 4> reserved_tail;
};
static_assert(sizeof(NativeImagePreference) == 0x40);

struct NativeImageLayout {
    u32 width;
    u32 height;
    u32 bytes_per_pixel;
    u32 row_pitch;
    u32 aligned_height;
    u32 block_height_log2;
    u32 alignment;
    u32 layer_stride;
    u64 size;
};

Result CalculateNativeImageLayout(NativeImageLayout* out, const NativeImagePreference& preference,
                                  u32 layers);

} // namespace Service::GSV
