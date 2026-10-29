// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <span>
#include <unordered_map>
#include "common/common_types.h"

namespace Tegra::Control {

// Encodings: NVIDIA classes/compute/clc7c0.h and clc7c0qmd.h (QMD V03.00).
// The owning GPU context serializes requests and scopes addresses to its address space.
class NativeQmdScheduler {
public:
    using Descriptor = std::array<u32, 64>;
    static constexpr size_t MaxDescriptors = 4096;
    static constexpr size_t MaxChain = 256;
    static constexpr s64 CounterLimit = 0xffff;

    struct Release {
        GPUVAddr address;
        u64 payload;
        u32 flags;

        u32 SizeBytes() const {
            const u32 format = flags >> 30;
            return format == 0 ? 16 : format == 1 ? 4 : 8;
        }
    };

    struct Host {
        // Dispatch/release callbacks record ordered GPU work; they do not signal its completion.
        void* context{};
        bool (*read)(void*, GPUVAddr, Descriptor&){};
        bool (*validate_release)(void*, const Release&){};
        bool (*dispatch)(void*, const Descriptor&){};
        bool (*release)(void*, const Release&){};
        bool (*should_stop)(void*){};
    };

    enum class Error {
        None,
        InvalidRequest,
        UnsupportedAction,
        DescriptorLimit,
        ReadFault,
        UnsupportedDescriptor,
        CounterOverflow,
        CounterUnderflow,
        UnsupportedRelease,
        InvalidRelease,
        DispatchFailed,
        ReleaseFailed,
        ChainLimit,
        Cancelled,
    };

    struct Result {
        Error error{};
        GPUVAddr address{};
        u32 dispatches{};
        u32 releases{};

        bool Succeeded() const {
            return error == Error::None;
        }
    };

    Result Signal(GPUVAddr address, u32 argument, const Host& host);
    bool HasPendingDispatch() const;
    bool HasFailed() const;

private:
    struct State {
        Descriptor descriptor{};
        s64 dependencies{};
        s64 deferred{};
        bool loaded{};
        bool scheduled{};
        bool issued{};
    };

    std::unordered_map<GPUVAddr, State> states;
    Result failure;
};

} // namespace Tegra::Control
