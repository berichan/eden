// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstring>
#include <utility>
#include "shader_recompiler/frontend/ampere/graphics_header.h"

namespace Shader::Ampere {

GraphicsHeader::Result GraphicsHeader::Decode(std::span<const u32> words, Stage stage) {
    if (words.size() != Words{}.size())
        return {.error = Error::WrongSize};
    if (stage != Stage::Vertex && stage != Stage::Fragment)
        return {.error = Error::UnsupportedStage};
    // NVIDIA clc797sph.h, SPHV4T1/T2; Mesa NAK's SPHv4 encoder.
    const bool vertex = stage == Stage::Vertex;
    if ((words[0] & 31) != (vertex ? 1U : 2U) || ((words[0] >> 10) & 15) != (vertex ? 1U : 5U))
        return {.error = Error::TypeMismatch};
    if (((words[0] >> 5) & 31) != 4)
        return {.error = Error::UnsupportedVersion};
    // Mesa emits tag 1; checked C797 producer captures also use tag 6. The
    // header tag does not replace instruction-by-instruction native validation.
    const u32 sass = (words[0] >> 17) & 15;
    if (sass != 1 && sass != 6)
        return {.error = Error::UnsupportedIsaVersion};
    // Native HIGH composition is unverified against the legacy HIGH<<24 contract.
    // Support the unambiguous LOW-only size until larger allocations are modeled.
    if (words[2] & 0xffffff)
        return {.error = Error::LocalMemoryHigh, .unsupported_words = 1U << 2};
    if (words[3] & 0xffffff)
        return {.error = Error::ReservedCrs, .unsupported_words = 1U << 3};

    GraphicsHeader decoded{};
    std::copy(words.begin(), words.end(), decoded.source.begin());
    decoded.local_memory_size = words[1] & 0xffffff;
    std::array<u32, sizeof(Shader::ProgramHeader) / sizeof(u32)> synthetic{};
    // Common metadata/resource hints have compatible positions. Native CRS and
    // HIGH never enter the legacy local-size accessor.
    synthetic[0] = words[0];
    synthetic[1] = words[1];
    synthetic[2] = words[2] & 0xff000000;
    synthetic[3] = words[3] & 0xff000000;
    synthetic[4] = words[4];
    const auto copy = [&](u32 first, u32 last) {
        for (u32 bit = first; bit <= last; ++bit)
            synthetic[bit / 32] |= words[bit / 32] & (1U << (bit % 32));
    };
    u32 unsupported{};
    const auto reject = [&](u32 first, u32 last) {
        for (u32 bit = first; bit <= last; ++bit)
            if (words[bit / 32] & (1U << (bit % 32)))
                unsupported |= 1U << (bit / 32);
    };
    reject(183, 183); // VPRS index
    if (vertex) {
        copy(184, 319); // System B and 32 generic input masks
        copy(336, 346); // Clip distances and point/fog inputs
        copy(348, 351); // Tessellation coordinates, instance/vertex IDs
        copy(424, 559); // System B and 32 generic output masks
        copy(576, 586);
        copy(588, 591);
        reject(320, 335); // Legacy color inputs
        reject(352, 399); // Fixed-function texture, viewport/barycentric inputs
        reject(423, 423); // VPRS output
        reject(560, 575); // Legacy color outputs
        reject(592, 879); // Fixed-function/viewport outputs and extended VMAP
    } else {
        copy(184, 447);   // System B and per-component generic interpolation
        copy(576, 609);   // Eight RGBA outputs, sample mask and depth
        reject(448, 575); // Legacy, viewport and barycentric interpolation
        reject(610, 611); // Interlock ticket and inner coverage
        reject(640, 871); // Per-vertex/EXT maps
    }
    if (unsupported)
        return {.error = Error::UnsupportedSemantics, .unsupported_words = unsupported};
    std::memcpy(&decoded.metadata, synthetic.data(), sizeof(decoded.metadata));
    return {.header = std::move(decoded)};
}

} // namespace Shader::Ampere
