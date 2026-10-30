// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <span>
#include <vector>
#include "common/common_types.h"

namespace Tegra::Control {

class NativeInlineUpload {
public:
    static constexpr u32 MaxBytes = 16U << 20;
    struct Parameters {
        u32 line_bytes;
        u32 line_count;
        u32 address_upper;
        u32 address_lower;
        u32 pitch;
    };
    struct Host {
        void* context;
        bool (*is_writable)(void*, GPUVAddr, size_t);
        bool (*wait_for_idle)(void*);
        bool (*write_checked)(void*, GPUVAddr, std::span<const u8>);
        bool (*should_stop)(void*);
    };
    enum class Error {
        None,
        UnsupportedLaunch,
        InvalidParameters,
        TransferPending,
        UnexpectedData,
        MemoryFault,
        WaitFailed,
        Cancelled,
    };

    Error Launch(const Parameters& parameters, u32 flags, const Host& host);
    Error PushWord(u32 word, const Host& host);
    bool IsPending() const;
    Error GetError() const;

private:
    Error Fail(Error failure);
    std::vector<u8> data;
    GPUVAddr address{};
    size_t received{};
    bool pending{};
    bool flush_only{};
    Error error{};
};

} // namespace Tegra::Control
