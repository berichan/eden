// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "video_core/control/native_report_semaphore.h"

namespace Tegra::Control {

NativeReportSemaphore::Error NativeReportSemaphore::Release(u32 upper, u32 lower, u32 payload,
                                                            u32 operation, const Host& host,
                                                            Engine engine) {
    // C797: RELEASE after all preceding writes, ALL pipeline. C7C0: plain RELEASE.
    // Both supported encodings use a ONE_WORD structure and a real host drain.
    // Reports, timestamps, reductions, acquire and four-word structures need other adapters.
    if ((engine != Engine::Graphics && engine != Engine::Compute) ||
        operation != (engine == Engine::Graphics ? 0x1000f010U : 0x10000000U)) {
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

NativeReportSemaphore::Error NativeReportSemaphore::ReportTimestamp(u32 upper, u32 lower,
                                                                    u32 payload, u32 operation,
                                                                    const TimestampHost& host) {
    // C797 REPORT_ONLY/NONE, ALL pipeline, FOUR_WORDS, FLUSH_DISABLE. The verified
    // SDK timestamp query sets C=0 and consumes the second u64 of the report.
    // Other report values and counter selections require their own semantics.
    if (operation != 0xf006 || payload != 0)
        return Error::UnsupportedOperation;
    if (upper > 0xff || (lower & 15))
        return Error::InvalidAddress;
    const auto& common = host.common;
    const auto stopped = [&] { return common.should_stop && common.should_stop(common.context); };
    if (stopped())
        return Error::Cancelled;
    const GPUVAddr address = (GPUVAddr{upper} << 32) | lower;
    if (!common.is_writable || !common.is_writable(common.context, address, 16))
        return Error::MemoryFault;
    if (!common.wait_for_idle || !common.wait_for_idle(common.context))
        return Error::WaitFailed;
    if (stopped())
        return Error::Cancelled;
    if (!host.timestamp_ns || !host.write_checked)
        return Error::MemoryFault;
    const auto timestamp = host.timestamp_ns(common.context);
    if (stopped())
        return Error::Cancelled;
    if (!host.write_checked(common.context, address, timestamp))
        return stopped() ? Error::Cancelled : Error::MemoryFault;
    return Error::None;
}

} // namespace Tegra::Control
