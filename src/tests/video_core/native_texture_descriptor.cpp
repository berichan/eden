// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include "shader_recompiler/frontend/ir/value.h"
#include "video_core/control/native_texture_descriptor.h"

using namespace Tegra::Control;

TEST_CASE("Native ASTC base mip footprints use compressed block rows", "[native-shader]") {
    NativeDescriptorWords image{};
    image[0] = 0x40;
    for (u32 i = 0; i < 4; ++i)
        image[0] |= (2U << (7 + 3 * i)) | ((i + 2) << (19 + 3 * i));
    image[2] = 3U << 21;
    image[3] = 4U << 3;
    image[4] = 639U | (1U << 22) | (1U << 23);
    image[5] = 359U | (1U << 31);
    const auto texture = NativeTextureDescriptor::Decode(image);
    REQUIRE(texture);
    REQUIRE(texture->IsAstc4x4Unorm2D());
    REQUIRE_FALSE(texture->IsRgba8Unorm2D());
    NativeDescriptorWords state{};
    state[0] = 2U | (2U << 3) | (2U << 6) | (1U << 13);
    state[1] = 2U | (2U << 4) | (3U << 6);
    state[2] = 15U << 20;
    const auto sampler = NativeSamplerDescriptor::Decode(state);
    REQUIRE(sampler);
    const auto mip = SelectNativeSampledMip(*texture, *sampler);
    REQUIRE(mip);
    REQUIRE(mip->size_bytes == 0x50000);
    REQUIRE(mip->block_height == 4);
    auto small = *texture;
    small.width = 3;
    small.height = 5;
    const auto edge = SelectNativeSampledMip(small, *sampler);
    REQUIRE(edge);
    REQUIRE(edge->size_bytes == 512);
    REQUIRE(edge->block_height == 0);
    auto unsupported = *texture;
    unsupported.max_mip = unsupported.view_max_mip = 1;
    REQUIRE_FALSE(SelectNativeSampledMip(unsupported, *sampler));
    unsupported = *texture;
    unsupported.srgb = false;
    REQUIRE_FALSE(SelectNativeSampledMip(unsupported, *sampler));
    unsupported = *texture;
    unsupported.swizzle[0] = 4;
    REQUIRE_FALSE(SelectNativeSampledMip(unsupported, *sampler));
    unsupported = *texture;
    unsupported.address = (u64{1} << 40) - 512;
    REQUIRE_FALSE(SelectNativeSampledMip(unsupported, *sampler));
}

TEST_CASE("Native packed float color sampling preserves format and missing alpha",
          "[native-shader]") {
    NativeDescriptorWords image{};
    image[0] = 0x21;
    for (u32 i = 0; i < 4; ++i)
        image[0] |= (7U << (7 + 3 * i)) | ((i == 3 ? 7U : i + 2) << (19 + 3 * i));
    image[2] = 3U << 21;
    image[4] = 15U | (1U << 23);
    image[5] = 7U | (1U << 31);
    const auto texture = NativeTextureDescriptor::Decode(image);
    REQUIRE(texture);
    REQUIRE(texture->IsB10G11R11Float2D());
    REQUIRE_FALSE(texture->IsRgba8Unorm2D());
    REQUIRE_FALSE(texture->srgb);
    REQUIRE(texture->swizzle == std::array<u32, 4>{2, 3, 4, 7});
    NativeDescriptorWords state{};
    state[0] = 2U | (2U << 3) | (2U << 6) | (1U << 13);
    state[1] = 2U | (2U << 4) | (3U << 6);
    state[2] = 15U << 20;
    const auto sampler = NativeSamplerDescriptor::Decode(state);
    REQUIRE(sampler);
    const auto mip = SelectNativeSampledMip(*texture, *sampler);
    REQUIRE(mip);
    REQUIRE(mip->level == 0);
    REQUIRE(mip->width == 16);
    REQUIRE(mip->height == 8);
    REQUIRE(mip->size_bytes == 512);
    for (const auto [word, bit] : std::array<std::pair<u32, u32>, 7>{
             {{0, 7}, {0, 30}, {0, 31}, {3, 27}, {4, 22}, {4, 24}, {5, 16}}}) {
        auto changed = image;
        changed[word] ^= 1U << bit;
        const auto decoded = NativeTextureDescriptor::Decode(changed);
        REQUIRE(decoded);
        REQUIRE_FALSE(decoded->IsB10G11R11Float2D());
        REQUIRE_FALSE(SelectNativeSampledMip(*decoded, *sampler));
    }
    auto fractional = *sampler;
    fractional.min_lod = 128;
    REQUIRE(SelectNativeSampledMip(*texture, fractional));
    auto unequal_filters = *sampler;
    unequal_filters.mag_filter = 1;
    REQUIRE_FALSE(SelectNativeSampledMip(*texture, unequal_filters));
    auto multi_level = *texture;
    multi_level.max_mip = multi_level.view_max_mip = 1;
    REQUIRE_FALSE(SelectNativeSampledMip(multi_level, *sampler));
}

TEST_CASE("Native color views accept padding only within the same byte pitch", "[native-shader]") {
    REQUIRE(NativeColorPitchMatches(40, 48));
    REQUIRE(NativeColorPitchMatches(20, 32));
    REQUIRE(NativeColorPitchMatches(1, 16));
    REQUIRE(NativeColorPitchMatches(16, 16));
    REQUIRE_FALSE(NativeColorPitchMatches(40, 49));
    REQUIRE_FALSE(NativeColorPitchMatches(40, 32));
    REQUIRE_FALSE(NativeColorPitchMatches(17, 16));
    REQUIRE_FALSE(NativeColorPitchMatches(0, 16));
}

TEST_CASE("Native narrowed views require ordinary normalized base-mip sampling state",
          "[native-shader]") {
    NativeDescriptorWords image{};
    image[0] = 8;
    for (u32 i = 0; i < 4; ++i)
        image[0] |= (2U << (7 + 3 * i)) | ((i + 2) << (19 + 3 * i));
    image[2] = (3U << 21) | (2U << 17);
    image[3] = 4U << 28;
    image[4] = 15U | (1U << 22) | (1U << 23);
    image[5] = 15U | (1U << 31);
    image[7] = 4U << 4;
    NativeDescriptorWords state{};
    state[0] = 2U | (2U << 3) | (2U << 6) | 0x26000;
    state[1] = 2U | (2U << 4) | (2U << 6) | (3U << 8);
    const auto texture = NativeTextureDescriptor::Decode(image);
    const auto sampler = NativeSamplerDescriptor::Decode(state);
    REQUIRE(texture);
    REQUIRE(sampler);
    REQUIRE(SelectNativeSampledMip(*texture, *sampler));
    for (const auto [word, bit] : std::array<std::pair<u32, u32>, 7>{
             {{1, 5}, {2, 17}, {3, 6}, {4, 22}, {5, 31}, {6, 6}, {7, 0}}}) {
        auto changed = image;
        changed[word] ^= 1U << bit;
        const auto decoded = NativeTextureDescriptor::Decode(changed);
        REQUIRE(decoded);
        REQUIRE_FALSE(SelectNativeSampledMip(*decoded, *sampler));
    }
    for (const auto [word, bit] : std::array<std::pair<u32, u32>, 8>{
             {{0, 9}, {0, 13}, {0, 20}, {1, 3}, {1, 10}, {1, 12}, {1, 25}, {2, 12}}}) {
        auto changed = state;
        changed[word] ^= 1U << bit;
        const auto decoded = NativeSamplerDescriptor::Decode(changed);
        REQUIRE(decoded);
        REQUIRE_FALSE(SelectNativeSampledMip(*texture, *decoded));
    }
}

TEST_CASE("Native selected-mip samples reject offsets and level-observing operations",
          "[native-shader]") {
    using Op = Shader::IR::Opcode;
    for (Op opcode : {Op::ImageSampleImplicitLod, Op::ImageSampleExplicitLod}) {
        Shader::IR::Inst sample{opcode, 0x20000};
        REQUIRE(IsNativeFixedMipSample(sample));
        sample.SetArg(3, Shader::IR::Value{u32{0}});
        REQUIRE_FALSE(IsNativeFixedMipSample(sample));
        sample.ClearArgs();
        sample.SetFlags(u32{0x120000});
        REQUIRE_FALSE(IsNativeFixedMipSample(sample));
    }
    for (Op opcode : {Op::ImageFetch, Op::ImageQueryDimensions, Op::ImageQueryLod, Op::ImageGather,
                      Op::ImageGradient}) {
        Shader::IR::Inst operation{opcode, 0x20000};
        REQUIRE_FALSE(IsNativeFixedMipSample(operation));
    }
}

TEST_CASE("Native fixed mip spans clamp tiling and retain 64-bit offsets", "[native-shader]") {
    NativeDescriptorWords image{};
    image[0] = 8;
    for (u32 i = 0; i < 4; ++i)
        image[0] |= (2U << (7 + 3 * i)) | ((i + 2) << (19 + 3 * i));
    image[2] = 3U << 21;
    image[3] = (4U << 3) | (9U << 28);
    image[4] = 639U | (1U << 22) | (1U << 23);
    image[5] = 359U | (1U << 31);
    image[7] = 9U << 4;
    auto texture = NativeTextureDescriptor::Decode(image);
    REQUIRE(texture);
    NativeDescriptorWords state{};
    state[0] = 2U | (2U << 3) | (2U << 6) | (1U << 13);
    state[1] = 2U | (2U << 4) | (2U << 6);
    auto sampler = NativeSamplerDescriptor::Decode(state);
    REQUIRE(sampler);
    constexpr std::array<u64, 10> offsets{0,        0xf0000,  0x140000, 0x154000, 0x159000,
                                          0x15a800, 0x15b000, 0x15b200, 0x15b400, 0x15b600};
    constexpr std::array<u32, 10> block_heights{4, 4, 4, 3, 2, 1, 0, 0, 0, 0};
    for (u32 level = 0; level < offsets.size(); ++level) {
        sampler->min_lod = sampler->max_lod = level << 8;
        const auto selected = SelectNativeSampledMip(*texture, *sampler);
        REQUIRE(selected);
        REQUIRE(selected->address == offsets[level]);
        REQUIRE(selected->block_height == block_heights[level]);
        REQUIRE(selected->level == level);
        REQUIRE(selected->size_bytes ==
                (level == 9 ? 0x15b800 - offsets[level] : offsets[level + 1] - offsets[level]));
    }
    sampler->min_lod = sampler->max_lod = 256;
    sampler->mip_filter = 1;
    REQUIRE_FALSE(SelectNativeSampledMip(*texture, *sampler));
    sampler->mip_filter = 2;
    sampler->min_lod = sampler->max_lod = 384;
    REQUIRE_FALSE(SelectNativeSampledMip(*texture, *sampler));
    sampler->min_lod = sampler->max_lod = 2560;
    REQUIRE_FALSE(SelectNativeSampledMip(*texture, *sampler));
    sampler->min_lod = sampler->max_lod = 0;
    texture->address = (u64{1} << 40) - 512;
    REQUIRE_FALSE(SelectNativeSampledMip(*texture, *sampler));
    texture->address = 0;
    texture->width = texture->height = 131072;
    texture->max_mip = texture->view_max_mip = 15;
    sampler->min_lod = sampler->max_lod = 256;
    const auto wide = SelectNativeSampledMip(*texture, *sampler);
    REQUIRE(wide);
    REQUIRE(wide->address == (u64{1} << 36));
    REQUIRE(wide->size_bytes == (u64{1} << 34));
    REQUIRE(wide->width == 65536);
}

TEST_CASE("Native descriptor pools check only the requested inclusive slot",
          "[ipc][native-shader]") {
    const auto texture = NativeDescriptorPool::Texture;
    const auto sampler = NativeDescriptorPool::Sampler;
    REQUIRE(NativeDescriptorAddress(texture, 1, 0x1000, 3, 3) == 0x100001060ULL);
    REQUIRE_FALSE(NativeDescriptorAddress(texture, 1, 0x1000, 3, 4));
    REQUIRE(NativeDescriptorAddress(texture, 0xff, 0xffffffe0, 0x3fffff, 0) == 0xffffffffe0ULL);
    REQUIRE_FALSE(NativeDescriptorAddress(texture, 0xff, 0xffffffe0, 1, 1));
    REQUIRE_FALSE(NativeDescriptorAddress(texture, 0x100, 0, 0, 0));
    REQUIRE_FALSE(NativeDescriptorAddress(texture, 0, 1, 0, 0));
    REQUIRE_FALSE(NativeDescriptorAddress(texture, 0, 0, 0x400000, 0));
    REQUIRE(NativeDescriptorAddress(sampler, 0, 0, 0xfffff, 0));
    REQUIRE_FALSE(NativeDescriptorAddress(sampler, 0, 0, 0x100000, 0));
}

TEST_CASE("Native blocklinear dimensions and address bits do not alias legacy layers",
          "[native-shader]") {
    NativeDescriptorWords words{};
    words[2] = 3U << 21;
    words[4] = 1U << 23;
    auto decoded = NativeTextureDescriptor::Decode(words);
    REQUIRE(decoded);
    REQUIRE(decoded->width == 1);
    REQUIRE(decoded->height == 1);
    REQUIRE(decoded->depth == 1);
    for (u32 bit = 9; bit < 49; ++bit) {
        auto address = words;
        address[1 + bit / 32] |= 1U << (bit % 32);
        const auto value = NativeTextureDescriptor::Decode(address);
        REQUIRE(value);
        REQUIRE(value->address == (u64{1} << bit));
        REQUIRE(value->reserved[2] == 0);
    }
    words[1] = 3U << 5;
    words[2] |= 2U << 17;
    words[4] |= 0x7ffff;
    words[5] = 0x3fffffff;
    decoded = NativeTextureDescriptor::Decode(words);
    REQUIRE(decoded->address == 0);
    REQUIRE(decoded->gob_depth_offset == 3);
    REQUIRE(decoded->reserved[2] == 0x40000);
    REQUIRE(decoded->width == 131072);
    REQUIRE(decoded->height == 131072);
    REQUIRE(decoded->depth == 32768);
    REQUIRE(decoded->words == words);
    words[2] = 2U << 21;
    REQUIRE_FALSE(NativeTextureDescriptor::Decode(words));
    words[2] = 3U << 21;
    words[3] = 6U << 3;
    REQUIRE_FALSE(NativeTextureDescriptor::Decode(words));
    words[3] = 0;
    words[7] = 1;
    REQUIRE_FALSE(NativeTextureDescriptor::Decode(words));
}

TEST_CASE("Native sampler metadata preserves reserved presets and decodes full filter fields",
          "[native-shader]") {
    NativeDescriptorWords words{};
    words[0] = 0x26000;
    words[1] = 2U | (2U << 4) | (2U << 6) | (3U << 8);
    const auto sampler = NativeSamplerDescriptor::Decode(words);
    REQUIRE(sampler);
    REQUIRE(sampler->reserved[0] == 0x24000);
    REQUIRE(sampler->srgb);
    REQUIRE_FALSE(sampler->force_unnormalized);
    REQUIRE(sampler->cube_interface == 3);
    REQUIRE(sampler->min_lod == 0);
    REQUIRE(sampler->max_lod == 0);
    REQUIRE(sampler->words == words);
    auto nearest_lod = words;
    nearest_lod[1] |= 8;
    REQUIRE(NativeSamplerDescriptor::Decode(nearest_lod)->integer_only_nearest_lod);
    REQUIRE(NativeSamplerDescriptor::Decode(nearest_lod)->reserved[1] == 0);
    auto signed_bias = words;
    signed_bias[1] |= 0x1fffU << 12;
    REQUIRE(NativeSamplerDescriptor::Decode(signed_bias)->lod_bias == -1);
    words[1] |= 4;
    REQUIRE_FALSE(NativeSamplerDescriptor::Decode(words));
    words[1] &= ~4U;
    words[0] |= 0x80000000;
    REQUIRE_FALSE(NativeSamplerDescriptor::Decode(words));
    words[0] &= ~0x80000000U;
    words[2] = 1;
    REQUIRE_FALSE(NativeSamplerDescriptor::Decode(words));
}
