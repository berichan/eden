// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstring>

#include "core/core.h"
#include "core/hle/kernel/k_event.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_readable_event.h"
#include "core/hle/kernel/svc_results.h"
#include "core/hle/service/gsv/display_session.h"
#include "core/hle/service/ipc_helpers.h"
#include "core/hle/service/sm/sm.h"
#include "core/hle/service/vi/container.h"
#include "core/hle/service/vi/manager_root_service.h"

namespace Service::GSV {
namespace {

constexpr Result ResultInvalidState{ErrorModule::VI, 5};
constexpr Result ResultInvalidArgument{ErrorModule::VI, 4};
constexpr Result ResultNotFound{ErrorModule::VI, 7};
constexpr Result ResultAlreadyRegistered{ErrorModule::VI, 9};

constexpr std::array<char, 8> DefaultDisplayName{'d', 'e', 'f', 'a', 'u', 'l', 't', '\0'};

bool HasInputWords(HLERequestContext& ctx, u32 words) {
    IPC::CommandHeader header{};
    std::memcpy(&header, ctx.CommandBuffer(), sizeof(header));
    return ctx.IsTipc() && words <= IPC::COMMAND_BUFFER_LENGTH && header.data_size == words &&
           ctx.GetDataPayloadOffset() <= IPC::COMMAND_BUFFER_LENGTH - words;
}

void Reply(HLERequestContext& ctx, Result result) {
    IPC::ResponseBuilder rb{ctx, 2};
    rb.Push(result);
}

} // namespace

DisplaySession::DisplaySession(Core::System& system, Kernel::KProcess* process)
    : ServiceFramework{system, "GsvDisplaySession"}, m_process{system.Kernel(), process},
      m_service_context{system, "GsvDisplaySession"} {
    static const FunctionInfo functions[] = {
        {1, &DisplaySession::GetMaxEventId, "GetMaxEventId"},
        {3, &DisplaySession::GetMaxSemaphoreId, "GetMaxSemaphoreId"},
        {11, &DisplaySession::InitializeEventQueue, "InitializeEventQueue"},
        {13, &DisplaySession::NotifyEvent, "NotifyEvent"},
        {21, &DisplaySession::ReadEventQueue, "ReadEventQueue"},
        {31, &DisplaySession::CreateSemaphore, "CreateSemaphore"},
        {32, &DisplaySession::DuplicateSemaphoreHandle, "DuplicateSemaphoreHandle"},
        {101, &DisplaySession::RegisterDisplay, "RegisterDisplay"},
        {102, &DisplaySession::UnregisterDisplay, "UnregisterDisplay"},
        {261, &DisplaySession::LinkVsyncSemaphore, "LinkVsyncSemaphore"},
        {262, &DisplaySession::UnlinkVsyncSemaphore, "UnlinkVsyncSemaphore"},
    };
    RegisterHandlersTipc(functions);
}

DisplaySession::~DisplaySession() {
    ReleaseDisplay();
    m_service_context.CloseEvent(m_notification_event);
    for (auto* event : m_semaphores) {
        m_service_context.CloseEvent(event);
    }
}

void DisplaySession::GetMaxEventId(HLERequestContext& ctx) {
    if (!HasInputWords(ctx, 0)) {
        Reply(ctx, Kernel::ResultInvalidSize);
        return;
    }
    IPC::ResponseBuilder rb{ctx, 3};
    rb.Push(ResultSuccess);
    rb.Push(DisplayEventQueue::MaxEventId);
}

void DisplaySession::GetMaxSemaphoreId(HLERequestContext& ctx) {
    if (!HasInputWords(ctx, 0)) {
        Reply(ctx, Kernel::ResultInvalidSize);
        return;
    }
    IPC::ResponseBuilder rb{ctx, 3};
    rb.Push(ResultSuccess);
    rb.Push<u32>(static_cast<u32>(m_semaphores.size() - 1));
}

void DisplaySession::InitializeEventQueue(HLERequestContext& ctx) {
    const auto result = [&]() -> Result {
        R_UNLESS(HasInputWords(ctx, 1), Kernel::ResultInvalidSize);
        IPC::RequestParser rp{ctx};
        const auto max_id = rp.Pop<u32>();
        auto queue = m_event_queue;
        R_TRY(queue.Initialize(max_id));
        auto* event = m_service_context.CreateEvent("DisplayNotifications");
        R_UNLESS(event != nullptr, Kernel::ResultOutOfResource);
        m_notification_event = event;
        m_event_queue = queue;
        ctx.AddCopyObject(&event->GetReadableEvent());
        LOG_DEBUG(Service, "Native display event queue initialized: max_id={}", max_id);
        R_SUCCEED();
    }();
    IPC::ResponseBuilder rb{ctx, 2, result.IsSuccess() ? 1U : 0U};
    rb.Push(result);
}

void DisplaySession::NotifyEvent(HLERequestContext& ctx) {
    if (!HasInputWords(ctx, 1)) {
        Reply(ctx, Kernel::ResultInvalidSize);
        return;
    }
    IPC::RequestParser rp{ctx};
    const auto id = rp.Pop<u32>();
    if (m_event_queue.Notify(id)) {
        m_notification_event->Signal(system.Kernel());
    }
    Reply(ctx, ResultSuccess);
}

void DisplaySession::ReadEventQueue(HLERequestContext& ctx) {
    const auto result = [&]() -> Result {
        R_UNLESS(HasInputWords(ctx, 0), Kernel::ResultInvalidSize);
        R_UNLESS(m_notification_event != nullptr, ResultInvalidState);
        const auto size = ctx.GetWriteBufferSize();
        R_UNLESS(size <= DisplayEventQueue::MaxBytes && ctx.CanWriteBuffer(),
                 Kernel::ResultInvalidSize);
        std::array<u8, DisplayEventQueue::MaxBytes> output{};
        m_event_queue.Drain(std::span{output}.first(size));
        ctx.WriteBuffer(output.data(), size);
        R_SUCCEED();
    }();
    Reply(ctx, result);
}

void DisplaySession::CreateSemaphore(HLERequestContext& ctx) {
    const auto result = [&]() -> Result {
        R_UNLESS(HasInputWords(ctx, 1), Kernel::ResultInvalidSize);
        IPC::RequestParser rp{ctx};
        const auto id = rp.Pop<u32>();
        R_UNLESS(id < m_semaphores.size(), ResultInvalidArgument);
        R_UNLESS(m_semaphores[id] == nullptr, ResultInvalidState);
        auto* event = m_service_context.CreateEvent("DisplaySemaphore");
        R_UNLESS(event != nullptr, Kernel::ResultOutOfResource);
        m_semaphores[id] = event;
        ctx.AddCopyObject(&event->GetReadableEvent());
        LOG_DEBUG(Service, "Native display semaphore created: id={}", id);
        R_SUCCEED();
    }();
    IPC::ResponseBuilder rb{ctx, 2, result.IsSuccess() ? 1U : 0U};
    rb.Push(result);
}

void DisplaySession::DuplicateSemaphoreHandle(HLERequestContext& ctx) {
    const auto result = [&]() -> Result {
        R_UNLESS(HasInputWords(ctx, 1), Kernel::ResultInvalidSize);
        IPC::RequestParser rp{ctx};
        const auto id = rp.Pop<u32>();
        R_UNLESS(id < m_semaphores.size(), ResultInvalidArgument);
        R_UNLESS(m_semaphores[id] != nullptr, ResultNotFound);
        auto object = ctx.GetObjectFromHandle<Kernel::KAutoObject>(ctx.GetCopyHandle(0));
        R_UNLESS(object.IsNotNull(), Kernel::ResultInvalidHandle);
        ctx.AddCopyObject(std::move(object));
        LOG_DEBUG(Service, "Native display semaphore handle duplicated: id={}", id);
        R_SUCCEED();
    }();
    IPC::ResponseBuilder rb{ctx, 2, result.IsSuccess() ? 1U : 0U};
    rb.Push(result);
}

void DisplaySession::RegisterDisplay(HLERequestContext& ctx) {
    const auto result = [&]() -> Result {
        R_UNLESS(HasInputWords(ctx, 3), Kernel::ResultInvalidSize);
        IPC::RequestParser rp{ctx};
        const auto name = rp.PopRaw<std::array<char, 8>>();
        const auto notification_id = rp.Pop<u32>();
        R_UNLESS((name != std::array<char, 8>{}) && notification_id != 0 &&
                     notification_id <= DisplayEventQueue::MaxEventId,
                 ResultInvalidArgument);
        R_UNLESS(name == DefaultDisplayName, ResultNotFound);
        R_UNLESS(!m_display, ResultAlreadyRegistered);
        if (!m_container) {
            const auto root =
                system.ServiceManager().GetService<VI::IManagerRootService>("vi:m", true);
            R_UNLESS(root != nullptr, Kernel::ResultInvalidState);
            m_container = root->GetContainer();
        }
        u64 display_id{};
        R_TRY(m_container->OpenDisplay(&display_id, VI::DisplayName{"Default"}));
        m_display = DisplayRegistration{display_id, notification_id};
        LOG_DEBUG(Service, "Native default display registered: id={}, notification_id={}",
                  display_id, notification_id);
        R_SUCCEED();
    }();
    Reply(ctx, result);
}

void DisplaySession::UnregisterDisplay(HLERequestContext& ctx) {
    if (!HasInputWords(ctx, 2)) {
        Reply(ctx, Kernel::ResultInvalidSize);
        return;
    }
    IPC::RequestParser rp{ctx};
    if (rp.PopRaw<std::array<char, 8>>() == DefaultDisplayName) {
        ReleaseDisplay();
    }
    Reply(ctx, ResultSuccess);
}

void DisplaySession::ReleaseDisplay() {
    if (m_display) {
        for (size_t id = 0; id < m_vsync_displays.size(); ++id) {
            if (m_vsync_displays[id] == m_display->display_id) {
                m_container->UnlinkVsyncEvent(m_display->display_id,
                                              &m_semaphores[id]->GetReadableEvent());
                m_vsync_displays[id].reset();
            }
        }
        m_container->CloseDisplay(m_display->display_id);
        m_display.reset();
    }
}

void DisplaySession::LinkVsyncSemaphore(HLERequestContext& ctx) {
    const auto result = [&]() -> Result {
        R_UNLESS(HasInputWords(ctx, 3), Kernel::ResultInvalidSize);
        IPC::RequestParser rp{ctx};
        const auto name = rp.PopRaw<std::array<char, 8>>();
        const auto id = rp.Pop<u32>();
        R_UNLESS((name != std::array<char, 8>{}) && id < m_semaphores.size(),
                 ResultInvalidArgument);
        R_UNLESS(name == DefaultDisplayName && m_display, ResultNotFound);
        R_UNLESS(m_semaphores[id] != nullptr, ResultNotFound);
        R_UNLESS(!m_vsync_displays[id], ResultInvalidState);
        // The compositor drives these events; registration itself does not signal completion.
        m_container->LinkVsyncEvent(m_display->display_id, &m_semaphores[id]->GetReadableEvent());
        m_vsync_displays[id] = m_display->display_id;
        LOG_DEBUG(Service, "Native display vsync semaphore linked: display_id={}, id={}",
                  m_display->display_id, id);
        R_SUCCEED();
    }();
    Reply(ctx, result);
}

void DisplaySession::UnlinkVsyncSemaphore(HLERequestContext& ctx) {
    if (!HasInputWords(ctx, 3)) {
        Reply(ctx, Kernel::ResultInvalidSize);
        return;
    }
    IPC::RequestParser rp{ctx};
    const auto name = rp.PopRaw<std::array<char, 8>>();
    const auto id = rp.Pop<u32>();
    if (name == DefaultDisplayName && m_display && id < m_vsync_displays.size() &&
        m_vsync_displays[id] == m_display->display_id) {
        m_container->UnlinkVsyncEvent(m_display->display_id, &m_semaphores[id]->GetReadableEvent());
        m_vsync_displays[id].reset();
    }
    Reply(ctx, ResultSuccess);
}

} // namespace Service::GSV
