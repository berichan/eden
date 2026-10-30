// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <optional>
#include "common/common_types.h"

namespace Tegra::Control {

struct NativeIndexedDrawRequest {
    enum class Format : u32 { U8, U16, U32 };
    GPUVAddr address;
    u64 size_bytes;
    Format format;
    u32 count;
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
