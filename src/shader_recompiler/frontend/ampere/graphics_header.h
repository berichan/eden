// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <span>
#include "common/common_types.h"
#include "shader_recompiler/program_header.h"

namespace Shader::Ampere {

struct GraphicsHeader {
    static constexpr size_t SizeBytes = 128;
    using Words = std::array<u32, SizeBytes / sizeof(u32)>;
    enum class Stage { Vertex, Fragment };
    enum class Error {
        None,
        WrongSize,
        UnsupportedStage,
        TypeMismatch,
        UnsupportedVersion,
        UnsupportedIsaVersion,
        LocalMemoryHigh,
        ReservedCrs,
        UnsupportedSemantics,
    };
    struct Result;

    // SPHv4 source identity includes all 128 bytes, even padding that does not
    // affect the translated metadata. Native code starts after these bytes.
    Words source;
    Shader::ProgramHeader metadata;
    u32 local_memory_size;

    static Result Decode(std::span<const u32> words, Stage stage);
};

struct GraphicsHeader::Result {
    Error error{};
    std::optional<GraphicsHeader> header;
    u32 unsupported_words{};
};

} // namespace Shader::Ampere
