// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstring>
#include <limits>
#include "common/swap.h"
#include "video_core/control/native_indexed_draw.h"

namespace Tegra::Control {

std::optional<NativeIndexBounds> NativeIndexBounds::Decode(NativeIndexedDrawRequest::Format format,
                                                           std::span<const u8> bytes,
                                                           std::stop_token stop) {
    const auto raw = static_cast<u32>(format);
    if (raw > 2 || bytes.empty() || bytes.size() % (1U << raw))
        return std::nullopt;
    NativeIndexBounds result{(std::numeric_limits<u32>::max)(), 0};
    for (size_t offset = 0; offset < bytes.size(); offset += 1U << raw) {
        if ((offset & 0xfff) == 0 && stop.stop_requested())
            return std::nullopt;
        u32 value{};
        if (format == NativeIndexedDrawRequest::Format::U8) {
            value = bytes[offset];
        } else if (format == NativeIndexedDrawRequest::Format::U16) {
            u16_le encoded;
            std::memcpy(&encoded, bytes.data() + offset, sizeof(encoded));
            value = encoded;
        } else {
            u32_le encoded;
            std::memcpy(&encoded, bytes.data() + offset, sizeof(encoded));
            value = encoded;
        }
        result.minimum = (std::min)(result.minimum, value);
        result.maximum = (std::max)(result.maximum, value);
    }
    return result;
}

std::optional<NativeVertexFetch> NativeVertexFetch::Decode(const Registers& regs,
                                                           u32 maximum_index) {
    if ((regs.attribute & 0xc0000060U) || ((regs.attribute >> 27) & 7) != 7 ||
        (regs.control & ~0x1fffU) || !(regs.control & 0x1000) || regs.instanced != 0 ||
        regs.address_upper > 0xff || regs.size_upper > 0xff)
        return std::nullopt;
    const u32 format = (regs.attribute >> 21) & 63;
    u32 width{};
    switch (format) {
    case 1:
        width = 16;
        break; // RGBA32
    case 2:
        width = 12;
        break; // RGB32
    case 3:
        width = 8;
        break; // RGBA16
    case 4:
        width = 8;
        break; // RG32
    case 5:
        width = 6;
        break; // RGB16
    case 15:
        width = 4;
        break; // RG16
    case 18:
        width = 4;
        break; // R32
    case 27:
        width = 2;
        break; // R16
    default:
        return std::nullopt;
    }
    const u32 stride = regs.control & 0xfff;
    const u32 offset = (regs.attribute >> 7) & 0x3fff;
    const u64 bytes = u64{maximum_index} * stride + offset + width;
    const auto address = (GPUVAddr{regs.address_upper} << 32) | regs.address_lower;
    const u64 ceiling = (u64{regs.size_upper} << 32) | regs.size_lower;
    if (bytes > ceiling || bytes > (u64{1} << 40) - address)
        return std::nullopt;
    return NativeVertexFetch{address, bytes, regs.attribute & 31, offset, stride, format};
}

std::optional<NativeIndexedDrawRequest> NativeIndexedDraw::Decode(const Registers& regs) {
    if (regs.address_upper > 0xff || regs.size_upper > 0xff || regs.format > 2 ||
        regs.base_vertex != 0 || regs.base_instance != 0 || regs.restart != 0)
        return std::nullopt;
    const u64 limit = (u64{regs.size_upper} << 32) | regs.size_lower;
    const GPUVAddr base = (GPUVAddr{regs.address_upper} << 32) | regs.address_lower;
    const u64 offset = u64{regs.first} << regs.format;
    const u64 bytes = u64{regs.count} << regs.format;
    constexpr u64 AddressLimit = u64{1} << 40;
    if (offset > limit || bytes > limit - offset || offset > AddressLimit - base ||
        bytes > AddressLimit - base - offset)
        return std::nullopt;
    return NativeIndexedDrawRequest{base + offset, bytes,
                                    static_cast<NativeIndexedDrawRequest::Format>(regs.format),
                                    regs.count};
}

std::optional<NativeDepthClip> NativeDepthClip::Decode(u32 control, u32 input_range,
                                                       u32 viewport_min_z, u32 viewport_max_z) {
    // NVIDIA clc797.h; Mesa NVK programs SET_VIEWPORT_Z_CLIP independently
    // from SET_VIEWPORT_CLIP_CONTROL.MIN_Z_ZERO_MAX_Z_ONE.
    if (input_range > 1 || (control & ~0x33c9fU) || ((control >> 11) & 7) != 3 ||
        ((control >> 1) & 3) != 2 || (control & 0x30000) || (control & 0x19) != 0x19 ||
        (viewport_min_z & 0x7fffffff) != 0 || viewport_max_z != 0x3f800000)
        return std::nullopt;
    return NativeDepthClip{input_range == 0};
}

NativeIndexedDraw::Error NativeIndexedDraw::Begin(u32 argument) {
    if (pending)
        return Error::NestedBegin;
    // Initially support triangles, FIRST primitive/instance IDs and a normal
    // BEGIN/END pair. Other defined topologies and split/iterate modes need adapters.
    if (argument != 4)
        return Error::UnsupportedBegin;
    pending = true;
    submitted = false;
    return Error::None;
}

NativeIndexedDraw::Error NativeIndexedDraw::Draw(const Registers& regs, const Host& host) {
    if (!pending)
        return Error::MissingBegin;
    if (submitted)
        return Error::RepeatedDraw;
    if (host.should_stop && host.should_stop(host.context))
        return Error::Cancelled;
    const auto request = Decode(regs);
    if (!request)
        return Error::InvalidIndexRange;
    if (!host.draw || !host.draw(host.context, *request))
        return Error::DrawFailed;
    if (host.should_stop && host.should_stop(host.context))
        return Error::Cancelled;
    submitted = true;
    return Error::None;
}

NativeIndexedDraw::Error NativeIndexedDraw::End(u32 argument) {
    if (!pending)
        return Error::MissingBegin;
    if (argument != 0)
        return Error::UnsupportedEnd;
    if (!submitted)
        return Error::MissingDraw;
    pending = false;
    submitted = false;
    return Error::None;
}

bool NativeIndexedDraw::IsPending() const {
    return pending;
}

bool NativeIndexedDraw::HasSubmitted() const {
    return submitted;
}

} // namespace Tegra::Control
