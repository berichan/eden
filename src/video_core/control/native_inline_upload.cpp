// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstring>
#include "video_core/control/native_inline_upload.h"

namespace Tegra::Control {

NativeInlineUpload::Error NativeInlineUpload::Fail(Error failure) {
    if (error == Error::None) {
        error = failure;
    }
    return error;
}

NativeInlineUpload::Error NativeInlineUpload::Launch(const Parameters& parameters, u32 flags,
                                                     const Host& host) {
    if (error != Error::None) {
        return error;
    }
    if (pending) {
        return Fail(Error::TransferPending);
    }
    // C797 pitch, completion NONE/FLUSH_ONLY, optional sysmembar disable and
    // semaphore structure size (inert without RELEASE_SEMAPHORE). No reductions.
    if (!(flags & 1) || (flags & ~0x1051U)) {
        return Fail(Error::UnsupportedLaunch);
    }
    // Multi-line packing/padding and blocklinear transfers require separate adapters.
    if (parameters.line_count != 1 || parameters.line_bytes == 0 ||
        parameters.line_bytes > MaxBytes || parameters.address_upper > 0xff) {
        return Fail(Error::InvalidParameters);
    }
    const GPUVAddr target = (GPUVAddr{parameters.address_upper} << 32) | parameters.address_lower;
    if (parameters.line_bytes > (u64{1} << 40) - target) {
        return Fail(Error::InvalidParameters);
    }
    if (host.should_stop && host.should_stop(host.context)) {
        return Fail(Error::Cancelled);
    }
    if (!host.is_writable || !host.is_writable(host.context, target, parameters.line_bytes)) {
        return Fail(Error::MemoryFault);
    }
    data.resize(parameters.line_bytes);
    address = target;
    received = 0;
    pending = true;
    flush_only = (flags & 0x10) != 0;
    return Error::None;
}

NativeInlineUpload::Error NativeInlineUpload::PushWord(u32 word, const Host& host) {
    if (error != Error::None) {
        return error;
    }
    if (!pending) {
        return Fail(Error::UnexpectedData);
    }
    if (host.should_stop && host.should_stop(host.context)) {
        return Fail(Error::Cancelled);
    }
    const size_t amount = (std::min)(sizeof(word), data.size() - received);
    std::memcpy(data.data() + received, &word, amount);
    received += amount;
    if (received != data.size()) {
        return Error::None;
    }
    // MME emissions each carry a packet-last flag. Only the transfer byte count
    // establishes a complete payload; never expose partially collected words.
    if (!host.is_writable || !host.is_writable(host.context, address, data.size())) {
        return Fail(Error::MemoryFault);
    }
    if (!host.wait_for_idle || !host.wait_for_idle(host.context)) {
        return Fail(Error::WaitFailed);
    }
    if (host.should_stop && host.should_stop(host.context)) {
        return Fail(Error::Cancelled);
    }
    if (!host.write_checked || !host.write_checked(host.context, address, data)) {
        return Fail(Error::MemoryFault);
    }
    if (flush_only && !host.wait_for_idle(host.context)) {
        return Fail(Error::WaitFailed);
    }
    if (host.should_stop && host.should_stop(host.context)) {
        return Fail(Error::Cancelled);
    }
    pending = false;
    data.clear();
    return Error::None;
}

bool NativeInlineUpload::IsPending() const {
    return pending;
}

NativeInlineUpload::Error NativeInlineUpload::GetError() const {
    return error;
}

} // namespace Tegra::Control
