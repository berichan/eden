// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "video_core/control/native_semaphore_acquire.h"

namespace Tegra::Control {
NativeSemaphoreAcquire::Error NativeSemaphoreAcquire::Check(u32 upper, u32 lower, u32 payload,
                                                            u32 operation, const Host& host) {
    // C76F circular ACQ_GEQ with ACQUIRE_SWITCH enabled, no additional fields.
    if (operation != 0x1004)
        return Error::UnsupportedOperation;
    if (upper > 0xff || (lower & 3))
        return Error::InvalidAddress;
    const auto stopped = [&] { return host.should_stop && host.should_stop(host.context); };
    if (stopped())
        return Error::Cancelled;
    const GPUVAddr address = (GPUVAddr{upper} << 32) | lower;
    if (!host.is_readable || !host.is_readable(host.context, address, sizeof(u32)))
        return Error::MemoryFault;
    if (!host.wait_for_idle || !host.wait_for_idle(host.context))
        return Error::WaitFailed;
    if (stopped())
        return Error::Cancelled;
    u32 value{};
    if (!host.read_checked || !host.read_checked(host.context, address, value))
        return Error::MemoryFault;
    if (stopped())
        return Error::Cancelled;
    return IsSatisfied(value, payload) ? Error::None : Error::NotReady;
}
} // namespace Tegra::Control
