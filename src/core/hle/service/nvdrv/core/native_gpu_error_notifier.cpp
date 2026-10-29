// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "core/core.h"
#include "core/hle/kernel/k_event.h"
#include "core/hle/service/kernel_helpers.h"
#include "core/hle/service/nvdrv/core/native_gpu_error_notifier.h"

namespace Service::Nvidia::NvCore {

NativeGpuErrorNotifier::~NativeGpuErrorNotifier() {
    if (event) {
        event->GetReadableEvent().Close(*kernel);
        event->Close(*kernel);
    }
}

void NativeGpuErrorNotifier::Configure(bool enabled_) {
    std::scoped_lock lock{mutex};
    enabled = enabled_;
    notification = {};
}

NativeGpuErrorNotifier::Notification NativeGpuErrorNotifier::Read() const {
    std::scoped_lock lock{mutex};
    return notification;
}

bool NativeGpuErrorNotifier::IsEnabled() const {
    std::scoped_lock lock{mutex};
    return enabled;
}

bool NativeGpuErrorNotifier::RecordError(u32 information, u64 timestamp) {
    Kernel::KEvent* signal_event;
    Kernel::KernelCore* event_kernel;
    {
        std::scoped_lock lock{mutex};
        if (notification.status == 0xffff) {
            return false;
        }
        notification.timestamp = timestamp;
        notification.information = information;
        notification.status = 0xffff;
        signal_event = enabled ? event : nullptr;
        event_kernel = kernel;
    }
    // Signaling takes the kernel scheduler lock; do not hold the notifier mutex across it.
    if (signal_event) {
        signal_event->Signal(*event_kernel);
    }
    return true;
}

Kernel::KEvent* NativeGpuErrorNotifier::GetEvent(Core::System& system) {
    {
        std::scoped_lock lock{mutex};
        if (event) {
            return event;
        }
    }
    KernelHelpers::ServiceContext context{system, "NativeGpuErrorNotifier"};
    auto* candidate = context.CreateEvent("NativeGpuChannelError");
    if (!candidate) {
        return nullptr;
    }
    Kernel::KEvent* selected;
    bool signal;
    {
        std::scoped_lock lock{mutex};
        // KEvent retains its owner process; the temporary service context can now be released.
        if (!event) {
            event = candidate;
            candidate = nullptr;
            kernel = &system.Kernel();
        }
        selected = event;
        signal = enabled && notification.status == 0xffff;
    }
    context.CloseEvent(candidate);
    if (signal) {
        selected->Signal(system.Kernel());
    }
    return selected;
}

} // namespace Service::Nvidia::NvCore
