// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include "video_core/control/native_texture_descriptor.h"

using namespace Tegra::Control;

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
