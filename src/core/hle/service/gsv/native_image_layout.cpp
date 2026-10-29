// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <limits>
#include "common/alignment.h"
#include "core/hle/kernel/svc_results.h"
#include "core/hle/service/gsv/native_image_layout.h"

namespace Service::GSV {

Result CalculateNativeImageLayout(NativeImageLayout* out, const NativeImagePreference& preference,
                                  u32 layers) {
    constexpr Result InvalidArgument{0x100f3};
    R_UNLESS(preference.type == 1,
             preference.type == 2 ? Kernel::ResultNotImplemented : InvalidArgument);
    const u32 options = preference.options;
    const u32 format = preference.format;
    R_UNLESS(
        preference.reserved == 0 && preference.reserved_0c == 0 &&
            std::ranges::all_of(preference.reserved_tail, [](u64 value) { return value == 0; }) &&
            format >= 1 && format <= 6 && (options & 0xff) >= 1 && (options & 0xff) <= 2 &&
            (options >> 24) == 0 && layers != 0,
        InvalidArgument);
    R_UNLESS(format < 5 || ((options >> 8) & 0xffff) < 0x4000, InvalidArgument);
    const u32 row_alignment_log2 = preference.row_alignment_log2;
    R_UNLESS(row_alignment_log2 >= 3 && row_alignment_log2 <= 8, InvalidArgument);

    // Extents are stored minus one. GOBs have 64-byte rows and eight rows; this exponent also
    // includes the GOB's three row bits, whereas the texture decoder's block height does not.
    const u32 packed_extent = preference.packed_extent;
    const u32 width = (packed_extent & 0xffff) + 1;
    const u32 height = (packed_extent >> 16) + 1;
    const u32 bytes_per_pixel = format < 5 ? 4 : 8;
    const u32 pixels_per_gob = 64 / bytes_per_pixel;
    const u64 pitch_pixels = preference.pitch_gobs != 0
                                 ? u64{preference.pitch_gobs} * pixels_per_gob
                                 : Common::AlignUp(u64{width}, pixels_per_gob);
    const u32 aligned_height = Common::AlignUp(height, u32{1} << row_alignment_log2);
    constexpr auto Max = std::numeric_limits<u32>::max();
    R_UNLESS(pitch_pixels <= Max && pitch_pixels * aligned_height <= Max, InvalidArgument);
    const u64 raw_size = pitch_pixels * aligned_height * bytes_per_pixel;
    R_UNLESS(raw_size <= Max, InvalidArgument);
    const u64 image_size = std::max<u64>(raw_size, 512);
    const u64 layer_stride = Common::AlignUp(image_size, u64{8192});
    R_UNLESS(layer_stride <= Max, InvalidArgument);

    *out = {.width = width,
            .height = height,
            .bytes_per_pixel = bytes_per_pixel,
            .row_pitch = static_cast<u32>(pitch_pixels * bytes_per_pixel),
            .aligned_height = aligned_height,
            .block_height_log2 = row_alignment_log2 - 3,
            .alignment = 4096,
            .layer_stride = static_cast<u32>(layer_stride),
            .size = (u64{layers} - 1) * layer_stride + image_size};
    R_SUCCEED();
}

} // namespace Service::GSV
