// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <optional>
#include "common/common_types.h"

namespace Shader::IR {
class Inst;
}

namespace Tegra::Control {

using NativeDescriptorWords = std::array<u32, 8>;

enum class NativeDescriptorPool { Texture, Sampler };
std::optional<GPUVAddr> NativeDescriptorAddress(NativeDescriptorPool pool, u32 upper, u32 lower,
                                                u32 limit, u32 index);

// Native C797 blocklinear header fields, not a legacy TICEntry. Decoding metadata
// does not establish backing, tiling compatibility or support for executing a sample.
struct NativeTextureDescriptor {
    NativeDescriptorWords words{}, reserved{};
    GPUVAddr address{};
    u32 format{}, type{}, width{}, height{}, depth{};
    std::array<u32, 4> component_types{}, swizzle{};
    u32 block_width{}, block_height{}, block_depth{}, tile_width{}, gob_depth_offset{};
    u32 max_mip{}, view_min_mip{}, view_max_mip{}, samples{};
    bool packed{}, srgb{}, normalized{}, gob3d{}, depth_texture{};

    static std::optional<NativeTextureDescriptor> Decode(const NativeDescriptorWords& words);
    bool IsRgba8Unorm2D() const;
    bool IsB10G11R11Float2D() const;
    bool IsB10G11R11Float3D() const;
    bool IsAstc4x4Unorm2D() const;
};

struct NativeSamplerDescriptor {
    NativeDescriptorWords words{}, reserved{};
    std::array<u32, 3> wrap{};
    std::array<float, 4> border{};
    u32 mag_filter{}, min_filter{}, mip_filter{}, cube_interface{}, reduction{}, anisotropy{};
    u32 min_lod{}, max_lod{}, depth_compare_func{};
    s32 lod_bias{};
    bool depth_compare{}, srgb{}, force_unnormalized{}, integer_only_nearest_lod{};

    static std::optional<NativeSamplerDescriptor> Decode(const NativeDescriptorWords& words);
};

struct NativeSampledMip {
    GPUVAddr address;
    u64 size_bytes;
    u32 width, height, block_height, level;
    u32 depth{1}, block_depth{};
};

// Only for ordinary sampled-image operations. Computes a fixed integral LOD span;
// execution still requires independent backing, layout and coherence validation.
std::optional<NativeSampledMip> SelectNativeSampledMip(const NativeTextureDescriptor& texture,
                                                       const NativeSamplerDescriptor& sampler);
bool IsNativeFixedMipSample(const Shader::IR::Inst& instruction);

bool NativeColorPitchMatches(u32 logical_width, u32 storage_width);

} // namespace Tegra::Control
