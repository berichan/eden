// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <mutex>
#include "common/common_types.h"

namespace Core {
class System;
}
namespace Kernel {
class KEvent;
class KernelCore;
} // namespace Kernel

namespace Service::Nvidia::NvCore {

class NativeGpuErrorNotifier {
public:
    struct Notification {
        u64 timestamp{};
        u32 information{};
        u16 info16{};
        u16 status{};
    };
    static_assert(sizeof(Notification) == 16);

    NativeGpuErrorNotifier() = default;
    ~NativeGpuErrorNotifier();

    void Configure(bool enabled);
    Notification Read() const;
    bool IsEnabled() const;
    // Accepts a native information code and the guest counter value at the fault.
    bool RecordError(u32 information, u64 timestamp);
    Kernel::KEvent* GetEvent(Core::System& system);

private:
    mutable std::mutex mutex;
    bool enabled{};
    Notification notification;
    Kernel::KernelCore* kernel{};
    Kernel::KEvent* event{};
};

} // namespace Service::Nvidia::NvCore
