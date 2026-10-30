// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <optional>
#include <span>
#include <stop_token>
#include "common/common_types.h"

namespace Tegra::Control {

struct NativeIndexedDrawRequest {
    enum class Format : u32 { U8, U16, U32 };
    GPUVAddr address;
    u64 size_bytes;
    Format format;
    u32 count;
};

struct NativeIndexBounds {
    u32 minimum, maximum;
    static std::optional<NativeIndexBounds> Decode(NativeIndexedDrawRequest::Format format,
                                                   std::span<const u8> bytes,
                                                   std::stop_token stop = {});
};

struct NativeVertexFetch {
    struct Registers {
        u32 attribute;
        u32 control, address_upper, address_lower, size_upper, size_lower, instanced;
    };
    GPUVAddr address;
    u64 size_bytes;
    u32 stream, offset, stride, component_format;
    // Initially active, noninstanced float attributes. The returned prefix is a
    // conservative upload range that preserves the original vertex indices.
    static std::optional<NativeVertexFetch> Decode(const Registers& regs, u32 maximum_index);
};

struct NativeDepthClip {
    bool negative_one_to_one;
    // XYZ primitive clipping, tight Z guardband and zero/one pixel clamps.
    // Input NDC range is an independent register.
    static std::optional<NativeDepthClip> Decode(u32 control, u32 input_range, u32 viewport_min_z,
                                                 u32 viewport_max_z);
};

class NativeIndexedDraw {
public:
    struct Registers {
        u32 address_upper, address_lower;
        u32 size_upper, size_lower;
        u32 format, first, count;
        u32 base_vertex, base_instance, restart;
    };
    struct Host {
        void* context{};
        bool (*draw)(void*, const NativeIndexedDrawRequest&){};
        bool (*should_stop)(void*){};
    };
    enum class Error {
        None,
        NestedBegin,
        UnsupportedBegin,
        MissingBegin,
        RepeatedDraw,
        InvalidIndexRange,
        DrawFailed,
        Cancelled,
        UnsupportedEnd,
        MissingDraw,
    };

    // C797 INDEX_BUFFER_SIZE is an access ceiling, not an allocation size or
    // end address. Only the requested byte span must fit the 40-bit GPUVA space.
    static std::optional<NativeIndexedDrawRequest> Decode(const Registers& regs);
    Error Begin(u32 argument);
    Error Draw(const Registers& regs, const Host& host);
    Error End(u32 argument);
    bool IsPending() const;
    bool HasSubmitted() const;

private:
    bool pending{};
    bool submitted{};
};

} // namespace Tegra::Control
