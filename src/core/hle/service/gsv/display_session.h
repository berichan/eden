// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <optional>
#include "core/hle/kernel/k_auto_object.h"
#include "core/hle/service/gsv/display_event_queue.h"
#include "core/hle/service/kernel_helpers.h"
#include "core/hle/service/service.h"

namespace Kernel {
class KProcess;
} // namespace Kernel

namespace Service::VI {
class Container;
}

namespace Service::GSV {

class DisplaySession final : public ServiceFramework<DisplaySession> {
public:
    explicit DisplaySession(Core::System& system, Kernel::KProcess* process);
    ~DisplaySession() override;

private:
    void GetMaxEventId(HLERequestContext& ctx);
    void GetMaxSemaphoreId(HLERequestContext& ctx);
    void GetMaxTimestampRecorderId(HLERequestContext& ctx);
    void ConfigureTimestampRecorderPool(HLERequestContext& ctx);
    void InitializeEventQueue(HLERequestContext& ctx);
    void NotifyEvent(HLERequestContext& ctx);
    void ReadEventQueue(HLERequestContext& ctx);
    void CreateSemaphore(HLERequestContext& ctx);
    void DuplicateSemaphoreHandle(HLERequestContext& ctx);
    void RegisterDisplay(HLERequestContext& ctx);
    void UnregisterDisplay(HLERequestContext& ctx);
    void SelectDisplayLayer(HLERequestContext& ctx);
    void LinkVsyncSemaphore(HLERequestContext& ctx);
    void UnlinkVsyncSemaphore(HLERequestContext& ctx);
    void ReleaseDisplay();
    void CreateLayer(HLERequestContext& ctx);
    void DestroyLayer(HLERequestContext& ctx);
    void QueryLayerBufferState(HLERequestContext& ctx);
    void BindLayer(HLERequestContext& ctx);
    void UnbindLayer(HLERequestContext& ctx);
    Result EnsureContainer();

    struct DisplayRegistration {
        u64 display_id;
        u32 notification_id;
    };

    struct LayerRegistration {
        u64 token;
        u64 layer_id;
        u32 state_notification_id;
        u32 buffer_notification_id;
    };

    Kernel::KScopedAutoObject<Kernel::KProcess> m_process;
    KernelHelpers::ServiceContext m_service_context;
    DisplayEventQueue m_event_queue;
    Kernel::KEvent* m_notification_event{};
    std::array<Kernel::KEvent*, 8> m_semaphores{};
    std::shared_ptr<VI::Container> m_container;
    std::optional<DisplayRegistration> m_display;
    std::array<std::optional<u64>, 8> m_vsync_displays{};
    std::array<std::optional<LayerRegistration>, 16> m_layers{};
    std::optional<u64> m_selected_layer;
    std::optional<u32> m_timestamp_recorder_count;
};

} // namespace Service::GSV
