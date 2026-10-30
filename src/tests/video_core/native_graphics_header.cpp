// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <catch2/catch_test_macros.hpp>
#include "shader_recompiler/frontend/ampere/graphics_header.h"

namespace {
using Header = Shader::Ampere::GraphicsHeader;
Header::Words Source(Header::Stage stage) {
    Header::Words words{};
    const bool vertex = stage == Header::Stage::Vertex;
    words[0] = (vertex ? 1U : 2U) | (4U << 5) | ((vertex ? 1U : 5U) << 10) | (1U << 17);
    return words;
}
void Set(Header::Words& words, u32 bit, u32 value = 1) {
    words[bit / 32] |= value << (bit % 32);
}
} // namespace

TEST_CASE("Native graphics headers validate identity and independent local memory semantics",
          "[native-shader]") {
    for (auto stage : {Header::Stage::Vertex, Header::Stage::Fragment}) {
        auto source = Source(stage);
        source[0] |= (1U << 25) | (1U << 26) | (1U << 27);
        source[1] = 0x72001230;
        source[2] = 0x03000000;
        source[3] = 0xa7000000;
        source[31] = 0x12345678; // Preserve nonsemantic tail bytes in source identity.
        const auto result = Header::Decode(source, stage);
        REQUIRE(result.header);
        REQUIRE(result.error == Header::Error::None);
        REQUIRE(Header::SizeBytes == 128);
        REQUIRE(sizeof(result.header->metadata) == 80);
        REQUIRE(result.header->source == source);
        REQUIRE(result.header->local_memory_size == 0x1230);
        REQUIRE(result.header->metadata.LocalMemorySize() == 0x1230);
        REQUIRE(result.header->metadata.common3.shader_local_memory_crs_size.Value() == 0);
        REQUIRE(result.header->metadata.common1.per_patch_attribute_count.Value() == 0x72);
        REQUIRE(result.header->metadata.common2.threads_per_input_primitive.Value() == 3);
        source[2] |= 1;
        REQUIRE(Header::Decode(source, stage).error == Header::Error::LocalMemoryHigh);
        source[2] &= 0xff000000;
        source[3] |= 1;
        REQUIRE(Header::Decode(source, stage).error == Header::Error::ReservedCrs);
    }
    const auto vertex = Source(Header::Stage::Vertex);
    REQUIRE(Header::Decode(std::span{vertex}.first(20), Header::Stage::Vertex).error ==
            Header::Error::WrongSize);
    REQUIRE(Header::Decode({}, Header::Stage::Vertex).error == Header::Error::WrongSize);
    REQUIRE(Header::Decode(vertex, static_cast<Header::Stage>(255)).error ==
            Header::Error::UnsupportedStage);
    REQUIRE(Header::Decode(vertex, Header::Stage::Fragment).error == Header::Error::TypeMismatch);
    auto bad = vertex;
    bad[0] ^= 1U << 5;
    REQUIRE(Header::Decode(bad, Header::Stage::Vertex).error == Header::Error::UnsupportedVersion);
    bad = vertex;
    bad[0] ^= 1U << 17;
    REQUIRE(Header::Decode(bad, Header::Stage::Vertex).error ==
            Header::Error::UnsupportedIsaVersion);
    for (auto stage : {Header::Stage::Vertex, Header::Stage::Fragment}) {
        for (u32 tag : {1U, 6U}) {
            auto native = Source(stage);
            native[0] = (native[0] & ~(15U << 17)) | (tag << 17);
            const auto result = Header::Decode(native, stage);
            REQUIRE(result.header);
            REQUIRE(result.header->metadata.common0.sass_version.Value() == tag);
        }
        for (u32 tag : {0U, 2U, 3U, 4U, 5U, 7U, 15U}) {
            auto native = Source(stage);
            native[0] = (native[0] & ~(15U << 17)) | (tag << 17);
            REQUIRE(Header::Decode(native, stage).error == Header::Error::UnsupportedIsaVersion);
        }
    }
}

TEST_CASE("Native vertex header input and output component masks retain their positions",
          "[native-shader]") {
    for (u32 index = 0; index < 32; ++index) {
        for (u32 component = 0; component < 4; ++component) {
            for (bool output : {false, true}) {
                auto source = Source(Header::Stage::Vertex);
                Set(source, (output ? 432 : 192) + index * 4 + component);
                const auto result = Header::Decode(source, Header::Stage::Vertex);
                REQUIRE(result.header);
                const auto mask = output ? result.header->metadata.vtg.OutputGeneric(index)
                                         : result.header->metadata.vtg.InputGeneric(index);
                for (u32 c = 0; c < 4; ++c)
                    REQUIRE(mask[c] == (c == component));
            }
        }
    }
    auto source = Source(Header::Stage::Vertex);
    Set(source, 184, 0xff);
    Set(source, 336, 0xff);
    Set(source, 350);
    Set(source, 351);
    Set(source, 424, 0xff);
    Set(source, 576, 0xff);
    const auto result = Header::Decode(source, Header::Stage::Vertex);
    REQUIRE(result.header);
    REQUIRE(result.header->metadata.vtg.imap_systemb.raw == 0xff);
    REQUIRE(result.header->metadata.vtg.omap_systemb.raw == 0xff);
    REQUIRE(result.header->metadata.vtg.clip_distances.Value() == 0xff);
    REQUIRE(result.header->metadata.vtg.omap_systemc.clip_distances.Value() == 0xff);
    REQUIRE(result.header->metadata.vtg.instance_id.Value() == 1);
    REQUIRE(result.header->metadata.vtg.vertex_id.Value() == 1);
}

TEST_CASE("Native fragment headers retain interpolation and every render target mask",
          "[native-shader]") {
    for (u32 index = 0; index < 32; ++index) {
        for (u32 component = 0; component < 4; ++component) {
            for (u32 mode = 1; mode < 4; ++mode) {
                auto source = Source(Header::Stage::Fragment);
                Set(source, 192 + index * 8 + component * 2, mode);
                const auto result = Header::Decode(source, Header::Stage::Fragment);
                REQUIRE(result.header);
                const auto map = result.header->metadata.ps.GenericInputMap(index);
                for (u32 c = 0; c < 4; ++c)
                    REQUIRE(map[c] == static_cast<Shader::PixelImap>(c == component ? mode : 0));
            }
        }
    }
    for (u32 target = 0; target < 8; ++target) {
        for (u32 component = 0; component < 4; ++component) {
            auto source = Source(Header::Stage::Fragment);
            Set(source, 576 + target * 4 + component);
            Set(source, 608);
            Set(source, 609);
            const auto result = Header::Decode(source, Header::Stage::Fragment);
            REQUIRE(result.header);
            const auto mask = result.header->metadata.ps.EnabledOutputComponents(target);
            for (u32 c = 0; c < 4; ++c)
                REQUIRE(mask[c] == (c == component));
            REQUIRE(result.header->metadata.ps.HasOutputComponents(target));
            REQUIRE_FALSE(result.header->metadata.ps.HasOutputComponents((target + 1) % 8));
            REQUIRE(result.header->metadata.ps.omap.sample_mask.Value() == 1);
            REQUIRE(result.header->metadata.ps.omap.depth.Value() == 1);
        }
    }
}

TEST_CASE("Native graphics header adaptation rejects unsupported semantic maps",
          "[native-shader]") {
    const auto check = [](Header::Stage stage, std::span<const u32> bits) {
        for (u32 bit : bits) {
            auto source = Source(stage);
            Set(source, bit);
            const auto result = Header::Decode(source, stage);
            REQUIRE_FALSE(result.header);
            REQUIRE(result.error == Header::Error::UnsupportedSemantics);
            REQUIRE(result.unsupported_words == (1U << (bit / 32)));
        }
    };
    check(Header::Stage::Vertex,
          std::array<u32, 20>{183, 320, 335, 352, 391, 392, 399, 423, 560, 575,
                              592, 631, 632, 639, 640, 671, 672, 799, 871, 879});
    check(Header::Stage::Fragment, std::array<u32, 16>{183, 448, 463, 464, 479, 480, 559, 560, 575,
                                                       610, 611, 640, 671, 672, 799, 871});
}
