// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <limits>
#include <catch2/catch_test_macros.hpp>
#include "core/hle/kernel/svc_results.h"
#include "core/hle/service/gsv/native_image_layout.h"
#include "video_core/textures/decoders.h"

namespace {
Service::GSV::NativeImagePreference Preference(u32 width, u32 height, u32 format = 1,
                                               u32 row_log2 = 7, u32 pitch = 0) {
    Service::GSV::NativeImagePreference result{};
    result.type = 1;
    result.packed_extent = (width - 1) | ((height - 1) << 16);
    result.row_alignment_log2 = row_log2;
    result.pitch_gobs = pitch;
    result.format = format;
    result.options = 1;
    return result;
}
} // namespace

TEST_CASE("Native image preferences account for GOB pitch and aligned rows", "[core][ipc]") {
    using namespace Service::GSV;
    const auto preference = Preference(2560, 1440, 1, 7, 160);
    NativeImageLayout layout{};
    REQUIRE(CalculateNativeImageLayout(&layout, preference, 1).IsSuccess());
    REQUIRE(layout.width == 2560);
    REQUIRE(layout.height == 1440);
    REQUIRE(layout.row_pitch == 10240);
    REQUIRE(layout.aligned_height == 1536);
    REQUIRE(layout.bytes_per_pixel == 4);
    REQUIRE(layout.block_height_log2 == 4);
    REQUIRE(layout.alignment == 4096);
    REQUIRE(layout.layer_stride == 0xf00000);
    REQUIRE(layout.size == 0xf00000);
    REQUIRE(CalculateNativeImageLayout(&layout, preference, 3).IsSuccess());
    REQUIRE(layout.size == 0x2d00000);
    auto alternate = preference;
    alternate.options = 2;
    REQUIRE(CalculateNativeImageLayout(&layout, alternate, 1).IsSuccess());
    REQUIRE(layout.size == 0xf00000);
    for (u32 format = 1; format <= 6; ++format) {
        REQUIRE(CalculateNativeImageLayout(&layout, Preference(1, 1, format, 3), 1).IsSuccess());
        REQUIRE(layout.bytes_per_pixel == (format < 5 ? 4U : 8U));
        REQUIRE(layout.size == 512);
        REQUIRE(layout.layer_stride == 8192);
        REQUIRE(CalculateNativeImageLayout(&layout, Preference(1, 1, format, 3), 3).IsSuccess());
        REQUIRE(layout.size == 2 * 8192 + 512);
    }
}

TEST_CASE("Automatic native GOB pitch agrees with the texture decoder geometry", "[core][ipc]") {
    using namespace Service::GSV;
    for (const u32 format : {1U, 5U}) {
        for (u32 block_height = 0; block_height <= 5; ++block_height) {
            for (const u32 width : {1U, 16U, 17U, 63U, 64U, 65U, 640U}) {
                for (const u32 height : {1U, 8U, 9U, 127U, 128U, 129U}) {
                    const auto preference = Preference(width, height, format, block_height + 3);
                    NativeImageLayout layout{};
                    REQUIRE(CalculateNativeImageLayout(&layout, preference, 1).IsSuccess());
                    REQUIRE(layout.size == Tegra::Texture::CalculateSize(true, format < 5 ? 4 : 8,
                                                                         width, height, 1,
                                                                         block_height, 0));
                }
            }
        }
    }
}

TEST_CASE("Native image validation rejects invalid fields without changing the output",
          "[core][ipc]") {
    using namespace Service::GSV;
    constexpr Result invalid{0x100f3};
    auto preference = Preference(2560, 1440);
    NativeImageLayout output{};
    output.size = 0x1122334455667788;
    const auto reject = [&](const NativeImagePreference& bad, u32 layers = 1) {
        REQUIRE(CalculateNativeImageLayout(&output, bad, layers) == invalid);
        REQUIRE(output.size == 0x1122334455667788);
    };
    for (const u32 format : {0U, 7U, 0xffffffffU}) {
        auto bad = preference;
        bad.format = format;
        reject(bad);
    }
    for (const u32 exponent : {0U, 2U, 9U, 256U}) {
        auto bad = preference;
        bad.row_alignment_log2 = exponent;
        reject(bad);
    }
    for (const auto field :
         {&NativeImagePreference::reserved, &NativeImagePreference::reserved_0c}) {
        auto bad = preference;
        bad.*field = 1;
        reject(bad);
    }
    for (size_t index = 0; index < preference.reserved_tail.size(); ++index) {
        auto bad = preference;
        bad.reserved_tail[index] = 1;
        reject(bad);
    }
    for (const u32 options : {0U, 3U, 0x1000000U}) {
        auto bad = preference;
        bad.options = options;
        reject(bad);
    }
    auto bad = preference;
    bad.format = 5;
    bad.options = 0x400001;
    reject(bad);
    reject(preference, 0);
    bad = preference;
    bad.pitch_gobs = 0xffffffff;
    reject(bad);
    reject(Preference(65536, 65536));
    reject(Preference(1, 1, 1, 3, 0x7fffff));
    bad = preference;
    bad.type = 0;
    reject(bad);
    bad.type = 2;
    REQUIRE(CalculateNativeImageLayout(&output, bad, 1) == Kernel::ResultNotImplemented);
    REQUIRE(output.size == 0x1122334455667788);
}

TEST_CASE("Native image layer accumulation preserves full-width size results", "[core][ipc]") {
    using namespace Service::GSV;
    const auto preference = Preference(1, 1, 1, 3, 0x7ffff0);
    NativeImageLayout layout{};
    REQUIRE(CalculateNativeImageLayout(&layout, preference, 2).IsSuccess());
    REQUIRE(layout.layer_stride == 0xffffe000);
    REQUIRE(layout.size == 0x1ffffc000);
    REQUIRE(CalculateNativeImageLayout(&layout, preference, std::numeric_limits<u32>::max())
                .IsSuccess());
    REQUIRE(layout.size == u64{0xffffe000} * std::numeric_limits<u32>::max());
}
