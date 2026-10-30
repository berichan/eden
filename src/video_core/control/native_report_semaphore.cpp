// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "video_core/control/native_report_semaphore.h"

namespace Tegra::Control {

NativeReportSemaphore::Error NativeReportSemaphore::Release(u32 upper, u32 lower, u32 payload,
                                                            u32 operation, const Host& host) {
    // C797: RELEASE, after all preceding writes, ALL pipeline, ONE_WORD structure.
    // Reports, timestamps, reductions, acquire and four-word structures need other adapters.
    if (operation != 0x1000f010) {
        return Error::UnsupportedOperation;
    }
    if (upper > 0xff || (lower & 3)) {
        return Error::InvalidAddress;
    }
    const auto stopped = [&] { return host.should_stop && host.should_stop(host.context); };
    if (stopped()) {
        return Error::Cancelled;
    }
    const GPUVAddr address = (GPUVAddr{upper} << 32) | lower;
    if (!host.is_writable || !host.is_writable(host.context, address, sizeof(payload))) {
        return Error::MemoryFault;
    }
    if (!host.wait_for_idle || !host.wait_for_idle(host.context)) {
        return Error::WaitFailed;
    }
    if (stopped()) {
        return Error::Cancelled;
    }
    if (!host.write_checked || !host.write_checked(host.context, address, payload)) {
        return Error::MemoryFault;
    }
    return Error::None;
}

} // namespace Tegra::Control
