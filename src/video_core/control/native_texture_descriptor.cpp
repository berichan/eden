// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <bit>
#include <cmath>
#include "shader_recompiler/frontend/ir/modifiers.h"
#include "shader_recompiler/frontend/ir/value.h"
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

bool NativeTextureDescriptor::IsB10G11R11Float2D() const {
    return format == 0x21 && (type == 1 || type == 7) && depth == 1 && samples == 0 && !packed &&
           !depth_texture && !srgb && component_types == std::array<u32, 4>{7, 7, 7, 7} &&
           swizzle == std::array<u32, 4>{2, 3, 4, 7};
}

bool NativeTextureDescriptor::IsB10G11R11Float3D() const {
    return format == 0x21 && type == 2 && samples == 0 && !packed && !depth_texture && !srgb &&
           component_types == std::array<u32, 4>{7, 7, 7, 7} &&
           swizzle == std::array<u32, 4>{2, 3, 4, 7};
}

bool NativeTextureDescriptor::IsAstc4x4Unorm2D() const {
    return format == 0x40 && (type == 1 || type == 7) && depth == 1 && samples == 0 && !packed &&
           !depth_texture && component_types == std::array<u32, 4>{2, 2, 2, 2} &&
           swizzle == std::array<u32, 4>{2, 3, 4, 5};
}

bool NativeTextureDescriptor::IsBc4UnormRrr1_2D() const {
    return format == 0x27 && (type == 1 || type == 7) && depth == 1 && samples == 0 && !packed &&
           !depth_texture && !srgb && component_types == std::array<u32, 4>{2, 2, 2, 2} &&
           swizzle == std::array<u32, 4>{2, 2, 2, 7};
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

std::optional<NativeSampledMip> SelectNativeSampledMip(const NativeTextureDescriptor& texture,
                                                       const NativeSamplerDescriptor& sampler) {
    const bool astc = texture.IsAstc4x4Unorm2D();
    const bool bc4 = texture.IsBc4UnormRrr1_2D();
    const bool compressed = astc || bc4;
    const bool volume = texture.IsB10G11R11Float3D();
    if (!(texture.IsRgba8Unorm2D() || (astc && texture.srgb) || texture.IsB10G11R11Float2D() ||
          volume || bc4) ||
        !texture.normalized || texture.block_width || (!volume && texture.block_depth) ||
        texture.tile_width || texture.gob3d || texture.gob_depth_offset || texture.view_min_mip ||
        !sampler.srgb || (texture.words[7] & 0xfff000) || (texture.words[6] & 0x387fffc0U) ||
        sampler.depth_compare || sampler.force_unnormalized || sampler.reduction ||
        sampler.anisotropy || sampler.lod_bias || sampler.integer_only_nearest_lod ||
        sampler.mag_filter != sampler.min_filter || (sampler.words[1] & 0x7c000000U) ||
        std::ranges::any_of(sampler.wrap, [](u32 mode) { return mode != 0 && mode != 2; }))
        return std::nullopt;
    // The SDK installs these registration/sampler presets. Preserve their raw
    // identity; they cannot establish a layout for an arbitrary uploaded image.
    for (u32 i = 0; i < texture.reserved.size(); ++i) {
        if (texture.reserved[i] && !(i == 2 && texture.reserved[i] == 0x40000))
            return std::nullopt;
        if (sampler.reserved[i] && !(i == 0 && sampler.reserved[i] == 0x24000))
            return std::nullopt;
    }
    // Ordinary samples cannot access another level when the header exposes only
    // mip zero, even with a ranged or fractional sampler clamp. Equal min/mag
    // filters above also preserve filtering after narrowing to the host base LOD.
    const bool single_mip = texture.view_max_mip == 0 && texture.max_mip == 0;
    if (!single_mip && (sampler.min_lod != sampler.max_lod || (sampler.min_lod & 255)))
        return std::nullopt;
    const u32 selected = single_mip ? 0 : sampler.min_lod >> 8;
    constexpr u64 AddressLimit = u64{1} << 40;
    if (selected > texture.view_max_mip || selected > texture.max_mip || selected > 15 ||
        (selected && (sampler.mip_filter == 1 || texture.type == 7)) ||
        texture.address >= AddressLimit || !texture.width || texture.width > 131072 ||
        !texture.height || texture.height > 131072 || texture.block_height > 5)
        return std::nullopt;
    // Uploaded compressed images currently cover a single base level. Color targets
    // continue to resolve independently established selected-mip cache images.
    if (compressed && !single_mip)
        return std::nullopt;
    if (volume) {
        // One XY GOB and one complete Z tile. Execution must independently
        // establish every canonical slice; this metadata is not a 3D cache
        // registration or permission to access overlapping padded volumes.
        if (!single_mip || texture.width > 16 || texture.height > 8 || texture.block_height ||
            texture.block_depth > 5 || texture.depth != (1U << texture.block_depth))
            return std::nullopt;
        const u64 size = u64{512} << texture.block_depth;
        if (size > AddressLimit - texture.address)
            return std::nullopt;
        return NativeSampledMip{
            texture.address,    size, texture.width, texture.height, 0, 0, texture.depth,
            texture.block_depth};
    }
    u64 offset{};
    for (u32 level = 0; level <= selected; ++level) {
        const u32 width = (std::max)(texture.width >> level, 1U);
        const u32 height = (std::max)(texture.height >> level, 1U);
        // Mesa NIL clamps tiling at every LOD before aligning that level. Use
        // 64-bit sizes: native dimension extensions can exceed legacy u32 helpers.
        u32 block_height = texture.block_height;
        const u64 block_width = compressed ? (u64{width} + 3) / 4 : width;
        const u64 block_rows = compressed ? (u64{height} + 3) / 4 : height;
        while (block_height && block_rows <= (4U << block_height))
            --block_height;
        const u64 tile_height = u64{8} << block_height;
        const u64 pitch = (block_width * (astc ? 16 : bc4 ? 8 : 4) + 63) & ~u64{63};
        const u64 aligned_height = (block_rows + tile_height - 1) & ~(tile_height - 1);
        const u64 size = pitch * aligned_height;
        if (offset > AddressLimit - texture.address ||
            size > AddressLimit - texture.address - offset)
            return std::nullopt;
        if (level == selected)
            return NativeSampledMip{
                texture.address + offset, size, width, height, block_height, level};
        offset += size;
    }
    return std::nullopt;
}

bool IsNativeFixedMipSample(const Shader::IR::Inst& instruction) {
    const auto opcode = instruction.GetOpcode();
    if (opcode != Shader::IR::Opcode::ImageSampleImplicitLod &&
        opcode != Shader::IR::Opcode::ImageSampleExplicitLod)
        return false;
    const auto flags = instruction.Flags<Shader::IR::TextureInstInfo>();
    return (flags.type == Shader::TextureType::Color2D ||
            flags.type == Shader::TextureType::Color3D) &&
           !(flags.raw & ~0x0047ffffU) && instruction.Arg(3).IsEmpty();
}

bool NativeColorPitchMatches(u32 logical_width, u32 storage_width) {
    return logical_width && storage_width >= logical_width &&
           ((u64{logical_width} * 4 + 63) & ~u64{63}) == ((u64{storage_width} * 4 + 63) & ~u64{63});
}

} // namespace Tegra::Control
