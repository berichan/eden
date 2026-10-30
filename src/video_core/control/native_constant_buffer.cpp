// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "video_core/control/native_constant_buffer.h"

namespace Tegra::Control {

NativeConstantBuffer::Error NativeConstantBuffer::Upload(Selection& selection,
                                                         std::span<const u32> words,
                                                         const Host& host) {
    // C797 selector/load fields: SIZE16:0, ADDRESS_UPPER7:0, LOAD_OFFSET15:0.
    if (selection.size > 0x1ffff || selection.address_upper > 0xff || selection.offset > 0xffff ||
        ((selection.address_lower | selection.offset) & 3)) {
        return Error::InvalidSelection;
    }
    const GPUVAddr base = (GPUVAddr{selection.address_upper} << 32) | selection.address_lower;
    if (selection.offset > selection.size ||
        words.size() > (selection.size - selection.offset) / sizeof(u32) ||
        selection.size > (u64{1} << 40) - base ||
        words.size() > (0x10000U - selection.offset) / sizeof(u32)) {
        return Error::OutOfBounds;
    }
    if (words.empty()) {
        return Error::None;
    }
    const auto stopped = [&] { return host.should_stop && host.should_stop(host.context); };
    if (stopped()) {
        return Error::Cancelled;
    }
    const GPUVAddr address = base + selection.offset;
    if (!host.is_writable || !host.is_writable(host.context, address, words.size_bytes())) {
        return Error::MemoryFault;
    }
    if (!host.wait_for_idle || !host.wait_for_idle(host.context)) {
        return Error::WaitFailed;
    }
    if (stopped()) {
        return Error::Cancelled;
    }
    if (!host.write_checked || !host.write_checked(host.context, address, words)) {
        return Error::MemoryFault;
    }
    // Keep an exhausted sentinel at 64KiB; do not infer hardware wrap semantics.
    // Another LOAD_OFFSET is required before further data or offset readback.
    selection.offset += static_cast<u32>(words.size_bytes());
    return Error::None;
}

} // namespace Tegra::Control
