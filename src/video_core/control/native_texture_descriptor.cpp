// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <bit>
#include <cmath>
#include "video_core/control/native_texture_descriptor.h"

namespace Tegra::Control {

std::optional<GPUVAddr> NativeDescriptorAddress(NativeDescriptorPool pool, u32 upper, u32 lower,
                                                u32 limit, u32 index) {
    if (pool != NativeDescriptorPool::Texture && pool != NativeDescriptorPool::Sampler)
        return std::nullopt;
    const u32 mask = pool == NativeDescriptorPool::Texture ? 0x3fffff : 0xfffff;
    if (upper > 0xff || (lower & 31) || (limit & ~mask) || index > limit)
        return std::nullopt;
    const GPUVAddr base = (GPUVAddr{upper} << 32) | lower;
    const u64 offset = u64{index} * sizeof(NativeDescriptorWords);
    constexpr u64 AddressLimit = u64{1} << 40;
    if (offset > AddressLimit - base ||
        sizeof(NativeDescriptorWords) > AddressLimit - base - offset)
        return std::nullopt;
    return base + offset;
}

std::optional<NativeTextureDescriptor> NativeTextureDescriptor::Decode(
    const NativeDescriptorWords& words) {
    // NVIDIA clc797tex.h, TEXHEAD_BL. Other header versions have different unions.
    if (((words[2] >> 21) & 7) != 3)
        return std::nullopt;
    NativeTextureDescriptor result;
    result.words = words;
    constexpr NativeDescriptorWords ReservedMasks{0, 0x19f,      0xe11e0000, 0xc200,
                                                  0, 0x40000000, 1,          0xff000000};
    for (u32 i = 0; i < words.size(); ++i)
        result.reserved[i] = words[i] & ReservedMasks[i];
    result.address = (GPUVAddr{words[2] & 0x1ffff} << 32) | (words[1] & 0xfffffe00U);
    result.format = words[0] & 127;
    for (u32 i = 0; i < 4; ++i) {
        result.component_types[i] = (words[0] >> (7 + 3 * i)) & 7;
        result.swizzle[i] = (words[0] >> (19 + 3 * i)) & 7;
    }
    result.packed = (words[0] >> 31) != 0;
    result.gob_depth_offset = (words[1] >> 5) & 3;
    result.block_width = words[3] & 7;
    result.block_height = (words[3] >> 3) & 7;
    result.block_depth = (words[3] >> 6) & 7;
    result.tile_width = (words[3] >> 10) & 7;
    result.gob3d = ((words[3] >> 13) & 1) != 0;
    result.depth_texture = ((words[3] >> 27) & 1) != 0;
    result.max_mip = words[3] >> 28;
    result.width = (words[4] & 0x1ffff) + 1;
    result.height = ((words[5] & 0xffff) | ((words[4] & 0x40000) >> 2)) + 1;
    result.depth = (((words[5] >> 16) & 0x3fff) | ((words[4] & 0x20000) >> 3)) + 1;
    result.srgb = ((words[4] >> 22) & 1) != 0;
    result.type = (words[4] >> 23) & 15;
    result.normalized = (words[5] >> 31) != 0;
    result.view_min_mip = words[7] & 15;
    result.view_max_mip = (words[7] >> 4) & 15;
    result.samples = (words[7] >> 8) & 15;
    if (result.block_width != 0 || result.block_height > 5 || result.block_depth > 5 ||
        result.tile_width > 5 || result.view_min_mip > result.view_max_mip ||
        result.view_max_mip > result.max_mip)
        return std::nullopt;
    return result;
}

bool NativeTextureDescriptor::IsRgba8Unorm2D() const {
    return format == 8 && (type == 1 || type == 7) && depth == 1 && samples == 0 && !packed &&
           !depth_texture && component_types == std::array<u32, 4>{2, 2, 2, 2} &&
           swizzle == std::array<u32, 4>{2, 3, 4, 5};
}

std::optional<NativeSamplerDescriptor> NativeSamplerDescriptor::Decode(
    const NativeDescriptorWords& words) {
    // C797's three-bit MAG_FILTER and two-bit cube enum differ from legacy TSCEntry.
    if (words[0] & 0x80000000U)
        return std::nullopt;
    NativeSamplerDescriptor result;
    result.words = words;
    result.reserved = {
        words[0] & 0x7f8fc000U, words[1] & 0x80000000U, 0, words[3] & 0xf0000fffU, 0, 0, 0, 0};
    for (u32 i = 0; i < 3; ++i)
        result.wrap[i] = (words[0] >> (3 * i)) & 7;
    for (u32 i = 0; i < 4; ++i) {
        result.border[i] = std::bit_cast<float>(words[4 + i]);
        if (!std::isfinite(result.border[i]))
            return std::nullopt;
    }
    result.depth_compare = ((words[0] >> 9) & 1) != 0;
    result.depth_compare_func = (words[0] >> 10) & 7;
    result.srgb = ((words[0] >> 13) & 1) != 0;
    result.anisotropy = (words[0] >> 20) & 7;
    result.mag_filter = words[1] & 7;
    result.integer_only_nearest_lod = ((words[1] >> 3) & 1) != 0;
    result.min_filter = (words[1] >> 4) & 3;
    result.mip_filter = (words[1] >> 6) & 3;
    result.cube_interface = (words[1] >> 8) & 3;
    result.reduction = (words[1] >> 10) & 3;
    const u32 bias = (words[1] >> 12) & 0x1fff;
    result.lod_bias = static_cast<s32>(bias ^ 0x1000) - 0x1000;
    result.force_unnormalized = ((words[1] >> 25) & 1) != 0;
    result.min_lod = words[2] & 0xfff;
    result.max_lod = (words[2] >> 12) & 0xfff;
    if (result.mag_filter < 1 || result.mag_filter > 2 || result.min_filter < 1 ||
        result.min_filter > 3 || result.mip_filter < 1 || result.mip_filter > 3 ||
        result.reduction > 2 || result.min_lod > result.max_lod)
        return std::nullopt;
    return result;
}

} // namespace Tegra::Control
