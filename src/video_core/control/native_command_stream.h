// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <span>
#include "common/common_types.h"

namespace Tegra::Control {

// One submission, including packets split across GPFIFO entries. Engine state and
// MME argument accumulation belong to the channel and outlive this decoder.
class NativeCommandStream {
public:
    static constexpr u64 MaxWords = 16 * 1024 * 1024;
    static constexpr size_t MaxEntries = 65536;
    struct Host {
        void* context{};
        bool (*can_read)(void*, GPUVAddr, size_t){};
        bool (*read)(void*, GPUVAddr, std::span<u32>){};
        bool (*method)(void*, u32 subchannel, u32 method, u32 value, bool last){};
        bool (*barrier)(void*){};
        bool (*should_stop)(void*){};
    };
    enum class Error {
        None,
        InvalidEntry,
        ReadFault,
        UnsupportedPacket,
        InvalidMethod,
        MethodFailed,
        BarrierFailed,
        Cancelled,
        IncompletePacket,
        WordLimit,
    };
    struct Result {
        Error error{};
        GPUVAddr address{};
        u64 words{};
        u64 methods{};
        u32 method{};
        u32 subchannel{};
        bool Succeeded() const {
            return error == Error::None;
        }
    };

    Result Execute(std::span<const u64> entries, const Host& host);
    Result Process(std::span<const u32> words, GPUVAddr address, const Host& host);
    Result Finish();

private:
    Result Fail(Error error, GPUVAddr address);
    bool Emit(u32 value, bool last, GPUVAddr address, const Host& host);
    Result result;
    u32 method{};
    u32 subchannel{};
    u32 remaining{};
    u32 mode{};
};

} // namespace Tegra::Control
