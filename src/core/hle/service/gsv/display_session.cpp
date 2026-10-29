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

namespace Service::GSV {
namespace {

constexpr Result ResultInvalidState{ErrorModule::VI, 5};
constexpr Result ResultInvalidArgument{ErrorModule::VI, 4};

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
    };
    RegisterHandlersTipc(functions);
}

DisplaySession::~DisplaySession() {
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

} // namespace Service::GSV
