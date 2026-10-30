// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "video_core/control/native_semaphore_release.h"

namespace Tegra::Control {
NativeSemaphoreRelease::Error NativeSemaphoreRelease::Release(u32 upper, u32 lower, u32 payload,
                                                              u32 operation, const Host& host) {
    // C76F RELEASE, WFI disabled, default 16-byte timestamped structure.
    if (operation != 0x100002)
        return Error::UnsupportedOperation;
    if (upper > 0xff || (lower & 15))
        return Error::InvalidAddress;
    const auto stopped = [&] { return host.should_stop && host.should_stop(host.context); };
    if (stopped())
        return Error::Cancelled;
    const GPUVAddr address = (GPUVAddr{upper} << 32) | lower;
    if (!host.is_writable || !host.is_writable(host.context, address, 16))
        return Error::MemoryFault;
    if (!host.wait_for_idle || !host.wait_for_idle(host.context))
        return Error::WaitFailed;
    if (stopped())
        return Error::Cancelled;
    if (!host.timestamp_ns || !host.write_checked)
        return Error::MemoryFault;
    const auto timestamp = host.timestamp_ns(host.context);
    if (stopped())
        return Error::Cancelled;
    if (!host.write_checked(host.context, address, payload, timestamp)) {
        return stopped() ? Error::Cancelled : Error::MemoryFault;
    }
    return Error::None;
}
} // namespace Tegra::Control
