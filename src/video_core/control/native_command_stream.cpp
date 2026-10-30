// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include "video_core/control/native_command_stream.h"

namespace Tegra::Control {

NativeCommandStream::Result NativeCommandStream::Fail(Error error, GPUVAddr address) {
    result.error = error;
    result.address = address;
    result.method = method;
    result.subchannel = subchannel;
    return result;
}

bool NativeCommandStream::Emit(u32 value, bool last, GPUVAddr address, const Host& host) {
    if (!host.method || !host.method(host.context, subchannel, method, value, last)) {
        result.value = value;
        Fail(Error::MethodFailed, address);
        return false;
    }
    ++result.methods;
    return true;
}

NativeCommandStream::Result NativeCommandStream::Process(std::span<const u32> words,
                                                         GPUVAddr address, const Host& host) {
    if (!result.Succeeded()) {
        return result;
    }
    if (address >= (u64{1} << 40) || (address & 3) ||
        words.size() > ((u64{1} << 40) - address) / sizeof(u32)) {
        return Fail(Error::InvalidEntry, address);
    }
    if (words.size() > MaxWords - result.words) {
        return Fail(Error::WordLimit, address);
    }
    for (const u32 word : words) {
        if (host.should_stop && host.should_stop(host.context)) {
            return Fail(Error::Cancelled, address);
        }
        ++result.words;
        if (remaining) {
            if (!Emit(word, remaining == 1, address, host)) {
                return result;
            }
            --remaining;
            if (mode == 1 || mode == 5) {
                ++method;
                if (mode == 5) {
                    mode = 3;
                }
            }
        } else if (word != 0) {
            method = word & 0xfff;
            subchannel = (word >> 13) & 7;
            mode = word >> 29;
            const u32 count = (word >> 16) & 0x1fff;
            if (word & (1U << 12)) {
                return Fail(Error::InvalidMethod, address);
            }
            if (mode == 4) {
                if (!Emit(count, true, address, host)) {
                    return result;
                }
            } else if (mode == 1 || mode == 3 || mode == 5) {
                if ((mode == 1 && count && count - 1 > 0xfff - method) ||
                    (mode == 5 && count > 1 && method == 0xfff)) {
                    return Fail(Error::InvalidMethod, address);
                }
                remaining = count;
            } else {
                return Fail(Error::UnsupportedPacket, address);
            }
        }
        address += sizeof(u32);
    }
    return result;
}

NativeCommandStream::Result NativeCommandStream::Execute(std::span<const u64> entries,
                                                         const Host& host) {
    if (!result.Succeeded()) {
        return result;
    }
    if (entries.size() > MaxEntries) {
        return Fail(Error::InvalidEntry, 0);
    }
    constexpr GPUVAddr AddressMask = (u64{1} << 40) - 1;
    u64 total = result.words;
    // Validate all descriptors and mapped ranges before executing the first method.
    for (const u64 entry : entries) {
        const GPUVAddr address = entry & AddressMask;
        const size_t count = (entry >> 42) & 0x1fffff;
        if (host.should_stop && host.should_stop(host.context)) {
            return Fail(Error::Cancelled, address);
        }
        if ((entry & (u64{1} << 40)) || (address & 3) || (!count && address) ||
            count > ((u64{1} << 40) - address) / sizeof(u32)) {
            return Fail(Error::InvalidEntry, address);
        }
        if (count > MaxWords - total) {
            return Fail(Error::WordLimit, address);
        }
        total += count;
        if (count &&
            (!host.can_read || !host.can_read(host.context, address, count * sizeof(u32)))) {
            return Fail(Error::ReadFault, address);
        }
    }
    std::array<u32, 256> buffer;
    for (const u64 entry : entries) {
        GPUVAddr address = entry & AddressMask;
        size_t count = (entry >> 42) & 0x1fffff;
        if (host.should_stop && host.should_stop(host.context)) {
            return Fail(Error::Cancelled, address);
        }
        if ((entry >> 63) && (!host.barrier || !host.barrier(host.context))) {
            return Fail(Error::BarrierFailed, address);
        }
        while (count) {
            const size_t size = std::min(count, buffer.size());
            const auto chunk = std::span{buffer}.first(size);
            if (host.should_stop && host.should_stop(host.context)) {
                return Fail(Error::Cancelled, address);
            }
            if (!host.read || !host.read(host.context, address, chunk)) {
                return Fail(Error::ReadFault, address);
            }
            if (!Process(chunk, address, host).Succeeded()) {
                return result;
            }
            count -= size;
            address += size * sizeof(u32);
        }
    }
    return Finish();
}

NativeCommandStream::Result NativeCommandStream::Finish() {
    if (result.Succeeded() && remaining) {
        return Fail(Error::IncompletePacket, 0);
    }
    return result;
}

} // namespace Tegra::Control
