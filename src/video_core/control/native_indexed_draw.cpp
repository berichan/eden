// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "video_core/control/native_indexed_draw.h"

namespace Tegra::Control {

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
