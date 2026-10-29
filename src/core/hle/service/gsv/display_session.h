// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include "core/hle/kernel/k_auto_object.h"
#include "core/hle/service/gsv/display_event_queue.h"
#include "core/hle/service/kernel_helpers.h"
#include "core/hle/service/service.h"

namespace Kernel {
class KProcess;
} // namespace Kernel

namespace Service::GSV {

class DisplaySession final : public ServiceFramework<DisplaySession> {
public:
    explicit DisplaySession(Core::System& system, Kernel::KProcess* process);
    ~DisplaySession() override;

private:
    void GetMaxEventId(HLERequestContext& ctx);
    void GetMaxSemaphoreId(HLERequestContext& ctx);
    void InitializeEventQueue(HLERequestContext& ctx);
    void NotifyEvent(HLERequestContext& ctx);
    void ReadEventQueue(HLERequestContext& ctx);
    void CreateSemaphore(HLERequestContext& ctx);

    Kernel::KScopedAutoObject<Kernel::KProcess> m_process;
    KernelHelpers::ServiceContext m_service_context;
    DisplayEventQueue m_event_queue;
    Kernel::KEvent* m_notification_event{};
    std::array<Kernel::KEvent*, 8> m_semaphores{};
};

} // namespace Service::GSV
