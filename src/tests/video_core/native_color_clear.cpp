// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include "core/core.h"
#include "core/device_memory.h"
#include "video_core/control/native_color_clear.h"
#include "video_core/control/native_gpu_context.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/gpu.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/memory_manager.h"
#include "video_core/renderer_null/null_rasterizer.h"
#include "video_core/texture_cache/util.h"
#include "video_core/textures/astc.h"
#include "video_core/textures/decoders.h"

namespace {
using Clear = Tegra::Control::NativeColorClear;
using Layout = Tegra::Texture::BlockLinearLayout;
Clear::Registers Registers() {
    Clear::Registers regs{};
    regs.argument = 0x3c;
    regs.target = {0x20, 0x100000, 2560, 1440, 0xe0, 0x40, 1, 0x3c0000, 0, 0};
    regs.color = {0x3f000000, 0x3f800000, 0x40000000, 0x3f800000};
    regs.control = 0x101;
    regs.scissor_enable = 1;
    regs.scissor_x = 2560U << 16;
    regs.scissor_y = 1440U << 16;
    return regs;
}
// Independent byte-coordinate form of Mesa NIL's TuringColor2D sector map.
u32 NativeOffset(u32 x, u32 y, u32 pitch_gobs, u32 block_height) {
    const u32 block_rows = 8U << block_height;
    return (y / block_rows) * pitch_gobs * (512U << block_height) +
           (x / 64) * (512U << block_height) + ((y % block_rows) / 8) * 512 + (x & 15) +
           ((y & 3) << 4) + ((x & 16) << 2) + ((y & 4) << 5) + ((x & 32) << 3);
}

VideoCommon::ImageInfo NativeTargetInfo(VideoCore::Surface::PixelFormat format) {
    VideoCommon::ImageInfo info;
    info.format = format;
    info.type = VideoCommon::ImageType::e2D;
    info.size = {16, 8, 1};
    info.layer_stride = info.maybe_unaligned_layer_stride = 512;
    info.layout = Layout::TuringColor2D;
    return info;
}
} // namespace

TEST_CASE("Native ASTC color sectors decode clipped edge blocks", "[native-shader]") {
    constexpr u32 width = 13, height = 9;
    std::vector<u8> tiled(512);
    for (u32 row = 0; row < 3; ++row) {
        for (u32 column = 0; column < 4; ++column) {
            const u8 red = static_cast<u8>(3 + 40 * column);
            const u8 green = static_cast<u8>(5 + 65 * row);
            // Standard ASTC LDR void-extent block with replicated 16-bit colors.
            const std::array<u8, 16> block{0xfc, 0xfd, 0xff,  0xff,  0xff, 0xff, 0xff, 0xff,
                                           red,  red,  green, green, 7,    7,    255,  255};
            for (u32 byte = 0; byte < block.size(); ++byte)
                tiled[NativeOffset(column * 16 + byte, row, 1, 0)] = block[byte];
        }
    }
    std::vector<u8> linear(4 * 3 * 16), pixels(width * height * 4);
    Tegra::Texture::UnswizzleTexture(linear, tiled, 16, 4, 3, 1, 0, 0, 1, Layout::TuringColor2D);
    Tegra::Texture::ASTC::Decompress(linear, width, height, 1, 4, 4, pixels);
    for (u32 y = 0; y < height; ++y) {
        for (u32 x = 0; x < width; ++x) {
            const auto pixel = (y * width + x) * 4;
            REQUIRE(pixels[pixel] == 3 + 40 * (x / 4));
            REQUIRE(pixels[pixel + 1] == 5 + 65 * (y / 4));
            REQUIRE(pixels[pixel + 2] == 7);
            REQUIRE(pixels[pixel + 3] == 255);
        }
    }
}

TEST_CASE("Native color clear decodes checked target and clipping state", "[ipc][video_core]") {
    auto regs = Registers();
    const auto clear = Clear::Decode(regs);
    REQUIRE(clear);
    REQUIRE(clear->address == 0x2000100000ULL);
    REQUIRE(clear->size_bytes == 0xf00000);
    REQUIRE(clear->width == 2560);
    REQUIRE(clear->height == 1440);
    REQUIRE(clear->block_height == 4);
    REQUIRE(clear->format == Clear::Format::B10G11R11Float);
    REQUIRE(clear->color == regs.color);
    REQUIRE(clear->extent_x == 2560);
    REQUIRE(clear->extent_y == 1440);
    regs.control = 0x1110;
    regs.rect_x = (200U << 16) | 20;
    regs.rect_y = (250U << 16) | 30;
    regs.scissor_x = (180U << 16) | 40;
    regs.scissor_y = (220U << 16) | 50;
    regs.viewport_x = (80U << 16) | 60;
    regs.viewport_y = (90U << 16) | 70;
    auto clipped = Clear::Decode(regs);
    REQUIRE(clipped);
    REQUIRE(clipped->x == 60);
    REQUIRE(clipped->y == 70);
    REQUIRE(clipped->extent_x == 80);
    REQUIRE(clipped->extent_y == 90);
    regs.rect_x = (10U << 16) | 200;
    REQUIRE(Clear::Decode(regs)->Empty());
    regs = Registers();
    regs.scissor_enable = 0;
    regs.scissor_x = 0;
    REQUIRE(Clear::Decode(regs)->extent_x == 2560);
    regs.target[7] = 0;
    REQUIRE(Clear::Decode(regs)->size_bytes == 0xf00000);
}

TEST_CASE("Native clear rejects unsupported surfaces and overflowing allocations",
          "[ipc][video_core]") {
    for (const auto [index, value] : std::array<std::pair<u32, u32>, 15>{{{0, 0x100},
                                                                          {1, 0x100001},
                                                                          {2, 0},
                                                                          {3, 0},
                                                                          {2, 0x10000000},
                                                                          {3, 0x20000},
                                                                          {4, 0xd7},
                                                                          {5, 0x41},
                                                                          {5, 0x60},
                                                                          {5, 0x1040},
                                                                          {6, 2},
                                                                          {8, 1},
                                                                          {9, 1},
                                                                          {2, 0xfffffff},
                                                                          {5, 0x10000}}}) {
        auto regs = Registers();
        regs.target[index] = value;
        REQUIRE_FALSE(Clear::Decode(regs));
    }
    for (u32 argument : {0x3dU, 0x1cU, 0x23cU, 0x43cU, 0x8000003cU}) {
        auto regs = Registers();
        regs.argument = argument;
        REQUIRE_FALSE(Clear::Decode(regs));
    }
    auto regs = Registers();
    regs.target[0] = 0xff;
    regs.target[1] = 0xfffffe00;
    REQUIRE_FALSE(Clear::Decode(regs));
    for (u32 field = 0; field < 5; ++field) {
        regs = Registers();
        switch (field) {
        case 0:
            regs.control |= 2;
            break;
        case 1:
            regs.scissor_enable = 2;
            break;
        case 2:
            regs.anti_alias = 1;
            break;
        case 3:
            regs.anti_alias_enable = 1;
            break;
        case 4:
            regs.anti_alias_raster = 1;
            break;
        }
        REQUIRE_FALSE(Clear::Decode(regs));
    }
}

TEST_CASE("Native color tiling matches byte sector coordinates and preserves padding",
          "[ipc][video_core]") {
    for (const auto [width, height, bh] :
         std::array<std::array<u32, 3>, 3>{{{16, 8, 0}, {17, 11, 1}, {33, 19, 1}}}) {
        const u32 row_bytes = width * 4;
        const u32 pitch_gobs = (row_bytes + 63) / 64;
        const u32 size = pitch_gobs * 64 * ((height + (8U << bh) - 1) & ~((8U << bh) - 1));
        std::vector<u8> linear(row_bytes * height);
        for (u32 i = 0; i < linear.size(); ++i)
            linear[i] = static_cast<u8>((i * 29 + i / 256) ^ (i >> 3));
        std::vector<u8> expected(size + 32, 0xcd), actual(size + 32, 0xcd);
        for (u32 y = 0; y < height; ++y)
            for (u32 x = 0; x < row_bytes; ++x)
                expected[16 + NativeOffset(x, y, pitch_gobs, bh)] = linear[y * row_bytes + x];
        Tegra::Texture::SwizzleTexture(std::span{actual}.subspan(16, size), linear, 4, width,
                                       height, 1, bh, 0, 1, Layout::TuringColor2D);
        REQUIRE(actual == expected);
        std::vector<u8> decoded(linear.size() + 32, 0xac);
        Tegra::Texture::UnswizzleTexture(std::span{decoded}.subspan(16, linear.size()),
                                         std::span{expected}.subspan(16, size), 4, width, height, 1,
                                         bh, 0, 1, Layout::TuringColor2D);
        REQUIRE(std::equal(linear.begin(), linear.end(), decoded.begin() + 16));
        REQUIRE(std::all_of(decoded.begin(), decoded.begin() + 16, [](u8 v) { return v == 0xac; }));
        REQUIRE(std::all_of(decoded.end() - 16, decoded.end(), [](u8 v) { return v == 0xac; }));
    }
    REQUIRE(NativeOffset(16, 0, 1, 0) == 0x40);
    REQUIRE(NativeOffset(0, 2, 1, 0) == 0x20);
    // Legacy default keeps the original sector ordering.
    std::array<u8, 512> source{}, legacy{}, native{};
    for (u32 i = 0; i < source.size(); ++i)
        source[i] = static_cast<u8>(i / 16);
    Tegra::Texture::SwizzleTexture(legacy, source, 1, 64, 8, 1, 0, 0);
    Tegra::Texture::SwizzleTexture(native, source, 1, 64, 8, 1, 0, 0, 1, Layout::TuringColor2D);
    REQUIRE(legacy[0x20] == source[16]);
    REQUIRE(legacy[0x40] == source[128]);
    REQUIRE(native[0x40] == source[16]);
    REQUIRE(native[0x20] == source[128]);
}

TEST_CASE("Single-layer native clears access the mip footprint independently of array pitch",
          "[ipc][video_core]") {
    auto regs = Registers();
    regs.target[2] = 640;
    regs.target[3] = 360;
    regs.target[4] = 0xd6;
    regs.color = {};
    for (u32 array_pitch : {0U, 1U, 0x56e00U, 0xffffffffU}) {
        regs.target[7] = array_pitch;
        const auto clear = Clear::Decode(regs);
        REQUIRE(clear);
        REQUIRE(clear->format == Clear::Format::RGBA8Srgb);
        REQUIRE(clear->size_bytes == 0xf0000);
        REQUIRE(clear->extent_x == 640);
        REQUIRE(clear->extent_y == 360);
    }
    regs.target = {0xff, 0xfffffe00, 16, 8, 0xe0, 0, 1, 0xffffffff, 0, 0};
    const auto boundary = Clear::Decode(regs);
    REQUIRE(boundary);
    REQUIRE(boundary->size_bytes == 512);
    REQUIRE(boundary->address + boundary->size_bytes == (u64{1} << 40));
    // Neither the argument nor target state can select another array layer.
    regs.argument |= 1U << 10;
    REQUIRE_FALSE(Clear::Decode(regs));
    regs.argument = 0x3c;
    regs.target[8] = 1;
    REQUIRE_FALSE(Clear::Decode(regs));
}

TEST_CASE("Native sRGB clears preserve float colors and guard unresolved conversion state",
          "[ipc][video_core]") {
    auto regs = Registers();
    regs.target[4] = 0xd6;
    regs.srgb_write = 1;
    const auto enabled = Clear::Decode(regs);
    REQUIRE(enabled);
    REQUIRE(enabled->format == Clear::Format::RGBA8Srgb);
    REQUIRE(enabled->color == regs.color);
    regs.srgb_write = 0;
    REQUIRE_FALSE(Clear::Decode(regs));
    regs.color = {0, 0x3f800000, 0, 0x3f000000};
    const auto endpoints = Clear::Decode(regs);
    REQUIRE(endpoints);
    REQUIRE(endpoints->color == regs.color);
    regs.srgb_write = 2;
    REQUIRE_FALSE(Clear::Decode(regs));
    regs.srgb_write = 1;
    regs.target[4] = 0xd7;
    REQUIRE_FALSE(Clear::Decode(regs));
}

TEST_CASE("Native UNORM clears preserve the captured format and clipping state",
          "[ipc][video_core]") {
    auto regs = Registers();
    regs.target[0] = 0x81;
    regs.target[1] = 0x0fa00000;
    regs.target[4] = 0xd5;
    regs.color = {};
    for (u32 srgb_write : {0U, 1U}) {
        regs.srgb_write = srgb_write;
        const auto clear = Clear::Decode(regs);
        REQUIRE(clear);
        REQUIRE(clear->format == Clear::Format::RGBA8Unorm);
        REQUIRE(clear->address == 0x810fa00000ULL);
        REQUIRE(clear->size_bytes == 0xf00000);
        REQUIRE(clear->color == regs.color);
        REQUIRE(clear->x == 0);
        REQUIRE(clear->y == 0);
        REQUIRE(clear->extent_x == 2560);
        REQUIRE(clear->extent_y == 1440);
    }
    regs.srgb_write = 2;
    REQUIRE_FALSE(Clear::Decode(regs));
    regs.srgb_write = 1;
    regs.control |= 0x10;
    regs.rect_x = (19U << 16) | 7;
    regs.rect_y = (23U << 16) | 11;
    const auto partial = Clear::Decode(regs);
    REQUIRE(partial);
    REQUIRE(partial->x == 7);
    REQUIRE(partial->y == 11);
    REQUIRE(partial->extent_x == 12);
    REQUIRE(partial->extent_y == 12);
    regs.color = {0x3f000000, 0x3e800000, 0x3f400000, 0x3f000000};
    REQUIRE_FALSE(Clear::Decode(regs));
    regs.srgb_write = 0;
    const auto linear = Clear::Decode(regs);
    REQUIRE(linear);
    REQUIRE(linear->color == regs.color);
    regs.srgb_write = 1;
    regs.color = {0x3f800000, 0, 0x3f800000, 0x3f000000};
    const auto endpoints = Clear::Decode(regs);
    REQUIRE(endpoints);
    REQUIRE(endpoints->color == regs.color);
}

TEST_CASE("Texture aliases must agree on native color layout", "[ipc][video_core]") {
    VideoCommon::ImageInfo legacy;
    legacy.type = VideoCommon::ImageType::e2D;
    legacy.format = VideoCore::Surface::PixelFormat::B10G11R11_FLOAT;
    legacy.size = {16, 8, 1};
    auto native = legacy;
    native.layout = Layout::TuringColor2D;
    for (u32 stride : {0U, 512U}) {
        legacy.layer_stride = native.layer_stride = stride;
        REQUIRE(VideoCommon::IsLayerStrideCompatible(legacy, legacy));
        REQUIRE_FALSE(VideoCommon::IsLayerStrideCompatible(legacy, native));
        REQUIRE_FALSE(VideoCommon::IsBlockLinearSizeCompatible(legacy, native, 0, 0, false));
        REQUIRE_FALSE(VideoCommon::IsBlockLinearSizeCompatibleBPPRelaxed(legacy, native, 0, 0));
        VideoCommon::ImageBase image{legacy, 0x100000, 0x200000};
        REQUIRE(VideoCommon::IsSubCopy(legacy, image, 0x100000));
        REQUIRE_FALSE(VideoCommon::IsSubCopy(native, image, 0x100000));
    }
}

TEST_CASE("Native color target identity and reformatting distinguish all supported format pairs",
          "[native-shader]") {
    using namespace VideoCommon;
    using Format = VideoCore::Surface::PixelFormat;
    constexpr std::array formats{Format::B10G11R11_FLOAT, Format::A8B8G8R8_SRGB,
                                 Format::A8B8G8R8_UNORM};
    constexpr GPUVAddr address = 0x2000100000ULL;
    constexpr DAddr backing = 0x100000;
    for (const auto old_format : formats) {
        for (const auto new_format : formats) {
            CAPTURE(old_format, new_format);
            ImageBase image{NativeTargetInfo(old_format), address, backing};
            const auto requested = NativeTargetInfo(new_format);
            REQUIRE(IsNativeColorTargetImage(requested, image, address, backing) ==
                    (old_format == new_format));
            REQUIRE(CanReformatNativeColorTargetImage(requested, image, address, backing) ==
                    (old_format != new_format));
            // CPU backing is authoritative and may be uploaded to the replacement.
            REQUIRE(True(image.flags & ImageFlagBits::CpuModified));
            image.flags |= ImageFlagBits::GpuModified;
            REQUIRE(IsNativeColorTargetImage(requested, image, address, backing) ==
                    (old_format == new_format));
            REQUIRE_FALSE(CanReformatNativeColorTargetImage(requested, image, address, backing));
        }
    }
}

TEST_CASE("Native color reformatting rejects aliases overlap and unsafe image states",
          "[native-shader]") {
    using namespace VideoCommon;
    using Format = VideoCore::Surface::PixelFormat;
    constexpr GPUVAddr address = 0x2000100000ULL;
    constexpr DAddr backing = 0x100000;
    const auto old = NativeTargetInfo(Format::B10G11R11_FLOAT);
    const auto requested = NativeTargetInfo(Format::A8B8G8R8_UNORM);
    for (const auto flag :
         {ImageFlagBits::Remapped, ImageFlagBits::Rescaled, ImageFlagBits::Sparse,
          ImageFlagBits::Converted, ImageFlagBits::AcceleratedUpload,
          ImageFlagBits::AsynchronousDecode, ImageFlagBits::IsDecoding, ImageFlagBits::Alias}) {
        CAPTURE(flag);
        ImageBase image{old, address, backing};
        image.flags |= flag;
        REQUIRE_FALSE(IsNativeColorTargetImage(old, image, address, backing));
        REQUIRE_FALSE(CanReformatNativeColorTargetImage(requested, image, address, backing));
    }
    ImageBase alias{old, address, backing};
    alias.aliased_images.emplace_back();
    REQUIRE_FALSE(IsNativeColorTargetImage(old, alias, address, backing));
    REQUIRE_FALSE(CanReformatNativeColorTargetImage(requested, alias, address, backing));
    ImageBase overlap{old, address, backing};
    overlap.overlapping_images.emplace_back();
    REQUIRE(IsNativeColorTargetImage(old, overlap, address, backing));
    REQUIRE_FALSE(CanReformatNativeColorTargetImage(requested, overlap, address, backing));
    ImageBase bad_overlap{old, address, backing};
    bad_overlap.flags |= ImageFlagBits::BadOverlap;
    REQUIRE_FALSE(CanReformatNativeColorTargetImage(requested, bad_overlap, address, backing));
}

TEST_CASE("Native color target predicates require canonical matching shapes and strides",
          "[native-shader]") {
    using namespace VideoCommon;
    using Format = VideoCore::Surface::PixelFormat;
    constexpr GPUVAddr address = 0x2000100000ULL;
    constexpr DAddr backing = 0x100000;
    const auto original = NativeTargetInfo(Format::B10G11R11_FLOAT);
    for (bool change_existing : {false, true}) {
        for (u32 variant = 0; variant < 23; ++variant) {
            CAPTURE(change_existing, variant);
            ImageBase image{original, address, backing};
            auto requested = original;
            auto& changed = change_existing ? image.info : requested;
            switch (variant) {
            case 0:
                changed.size.width = 17;
                break;
            case 1:
                changed.size.width = 15;
                break;
            case 2:
                changed.size.height = 7;
                break;
            case 3:
                changed.size.depth = 2;
                break;
            case 4:
                changed.block.width = 1;
                break;
            case 5:
                changed.block.height = 1;
                break;
            case 6:
                changed.block.height = 6;
                break;
            case 7:
                changed.block.depth = 1;
                break;
            case 8:
                changed.resources.levels = 2;
                break;
            case 9:
                changed.resources.layers = 2;
                break;
            case 10:
                changed.num_samples = 2;
                break;
            case 11:
                changed.type = ImageType::e3D;
                break;
            case 12:
                changed.layout = Layout::Tegra;
                break;
            case 13:
                changed.rescaleable = true;
                break;
            case 14:
                changed.is_sparse = true;
                break;
            case 15:
                changed.tile_width_spacing = 1;
                break;
            case 16:
                changed.layer_stride = 1024;
                break;
            case 17:
                changed.maybe_unaligned_layer_stride = 256;
                break;
            case 18:
                changed.format = Format::R8_UNORM;
                break;
            case 19:
                changed.size.width = 0;
                break;
            case 20:
                changed.size.height = 0;
                break;
            case 21:
                changed.size.width = 0xffffffff;
                break;
            default:
                changed.size.height = 0xffffffff;
                break;
            }
            REQUIRE_FALSE(IsNativeColorTargetImage(requested, image, address, backing));
            if (requested.format == original.format)
                requested.format = Format::A8B8G8R8_UNORM;
            REQUIRE_FALSE(CanReformatNativeColorTargetImage(requested, image, address, backing));
        }
    }
    for (u32 variant = 0; variant < 4; ++variant) {
        ImageBase image{original, address, backing};
        switch (variant) {
        case 0:
            image.gpu_addr += 512;
            break;
        case 1:
            image.cpu_addr += 512;
            break;
        case 2:
            ++image.cpu_addr_end;
            break;
        default:
            image.guest_size_bytes += 512;
            break;
        }
        REQUIRE_FALSE(IsNativeColorTargetImage(original, image, address, backing));
        auto requested = original;
        requested.format = Format::A8B8G8R8_UNORM;
        REQUIRE_FALSE(CanReformatNativeColorTargetImage(requested, image, address, backing));
    }
}

TEST_CASE("Native color identity permits only explicit same-pitch source cropping",
          "[native-shader]") {
    using namespace VideoCommon;
    using Format = VideoCore::Surface::PixelFormat;
    constexpr GPUVAddr address = 0x2000100000ULL;
    constexpr DAddr backing = 0x100000;
    auto storage = NativeTargetInfo(Format::B10G11R11_FLOAT);
    storage.size.width = 48;
    storage.layer_stride = storage.maybe_unaligned_layer_stride = 1536;
    ImageBase image{storage, address, backing};
    auto logical = storage;
    logical.size.width = 40;
    REQUIRE_FALSE(IsNativeColorTargetImage(logical, image, address, backing));
    REQUIRE(IsNativeColorTargetImage(logical, image, address, backing, true));
    logical.format = Format::A8B8G8R8_UNORM;
    REQUIRE_FALSE(CanReformatNativeColorTargetImage(logical, image, address, backing));
    logical = storage;
    logical.size.width = 49;
    REQUIRE_FALSE(IsNativeColorTargetImage(logical, image, address, backing, true));
    logical.size.width = 32;
    REQUIRE_FALSE(IsNativeColorTargetImage(logical, image, address, backing, true));
    logical = storage;
    logical.size.height = 7;
    REQUIRE_FALSE(IsNativeColorTargetImage(logical, image, address, backing, true));
    const GPUVAddr boundary = (u64{1} << 40) - 1536;
    ImageBase last{storage, boundary, backing};
    REQUIRE(IsNativeColorTargetImage(storage, last, boundary, backing));
    REQUIRE_FALSE(IsNativeColorTargetImage(storage, last, boundary + 1, backing));
}

TEST_CASE("Native clear failure cannot retire through the null rasterizer", "[ipc][video_core]") {
    Core::System system;
    Core::DeviceMemory memory;
    Tegra::MaxwellDeviceMemoryManager backing{memory};
    Tegra::GPU gpu{system, false, false};
    Null::RasterizerNull rasterizer{gpu};
    Tegra::MemoryManager space{system, backing, 32, 1ULL << 32, 16, 12};
    space.BindRasterizer(&rasterizer);
    Tegra::Engines::Maxwell3D engine{space};
    engine.BindRasterizer(&rasterizer);
    engine.SetNativeMacroState(std::make_shared<Tegra::Control::AmpereGraphicsState>());
    const auto regs = Registers();
    for (u32 i = 0; i < regs.target.size(); ++i)
        REQUIRE(engine.CallNativeMethod(system, 0x200 + i, regs.target[i], true, {}));
    REQUIRE(engine.CallNativeMethod(system, 0x43e, regs.control, true, {}));
    REQUIRE(engine.CallNativeMethod(system, 0x380, regs.scissor_enable, true, {}));
    REQUIRE(engine.CallNativeMethod(system, 0x381, regs.scissor_x, true, {}));
    REQUIRE(engine.CallNativeMethod(system, 0x382, regs.scissor_y, true, {}));
    REQUIRE_FALSE(engine.CallNativeMethod(system, 0x674, regs.argument, true, {}));
    REQUIRE(engine.HasNativeMacroFailure());
    REQUIRE(engine.GetRegisterValue(0x674) == 0);
    REQUIRE_FALSE(engine.CallNativeMethod(system, 0xd00, 9, true, {}));
    REQUIRE(engine.GetRegisterValue(0xd00) == 0);
}

TEST_CASE("Native clears honor unconditional override and guard global predicates",
          "[ipc][video_core]") {
    Core::System system;
    Core::DeviceMemory memory;
    Tegra::MaxwellDeviceMemoryManager backing{memory};
    Tegra::GPU gpu{system, false, false};
    Null::RasterizerNull rasterizer{gpu};
    Tegra::MemoryManager space{system, backing, 32, 1ULL << 32, 16, 12};
    space.BindRasterizer(&rasterizer);
    const auto regs = Registers();
    for (u32 render_override : {0U, 1U, 2U, 3U}) {
        Tegra::Engines::Maxwell3D engine{space};
        engine.BindRasterizer(&rasterizer);
        engine.SetNativeMacroState(std::make_shared<Tegra::Control::AmpereGraphicsState>());
        for (u32 i = 0; i < regs.target.size(); ++i)
            REQUIRE(engine.CallNativeMethod(system, 0x200 + i, regs.target[i], true, {}));
        REQUIRE(engine.CallNativeMethod(system, 0x651, render_override, true, {}));
        // Only NeverRender is a valid no-op on an unsupported renderer.
        REQUIRE(engine.CallNativeMethod(system, 0x674, regs.argument, true, {}) ==
                (render_override == 2));
        REQUIRE(engine.HasNativeMacroFailure() == (render_override != 2));
    }
    for (u32 mode : {0U, 1U, 2U, 3U, 4U, 5U}) {
        Tegra::Engines::Maxwell3D engine{space};
        engine.SetNativeMacroState(std::make_shared<Tegra::Control::AmpereGraphicsState>());
        REQUIRE(engine.CallNativeMethod(system, 0x4e, mode, true, {}) == (mode == 1));
        REQUIRE(engine.HasNativeMacroFailure() == (mode != 1));
    }
    for (u32 mode : {0U, 1U, 2U, 3U, 4U}) {
        Tegra::Engines::Maxwell3D engine{space};
        engine.SetNativeMacroState(std::make_shared<Tegra::Control::AmpereGraphicsState>());
        REQUIRE(engine.CallNativeMethod(system, 0x556, mode, true, {}) == (mode <= 1));
        REQUIRE(engine.HasNativeMacroFailure() == (mode > 1));
        if (mode <= 1)
            REQUIRE(engine.ShouldExecute() == (mode == 1));
    }
}
