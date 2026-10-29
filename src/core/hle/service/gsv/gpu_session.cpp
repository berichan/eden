// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstring>

#include "core/core.h"
#include "core/hle/kernel/k_client_session.h"
#include "core/hle/kernel/k_event.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_readable_event.h"
#include "core/hle/kernel/svc_results.h"
#include "core/hle/service/gsv/gpu_session.h"
#include "core/hle/service/ipc_helpers.h"

namespace Service::GSV {
namespace {

constexpr Result ResultAlreadyCreated{0x200f3};

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

// Keep a separate native endpoint ready before publishing its handle. Memory and mapping
// operations still need their own handlers; unregistered commands return protocol errors.
class KUserWorkerSession final : public ServiceFramework<KUserWorkerSession> {
public:
    explicit KUserWorkerSession(Core::System& system, Kernel::KProcess* process)
        : ServiceFramework{system, "GpuWorkerSession"}, m_process{system.Kernel(), process} {}
    ~KUserWorkerSession() override = default;

private:
    Kernel::KScopedAutoObject<Kernel::KProcess> m_process;
};

} // namespace

GpuSession::GpuSession(Core::System& system, Kernel::KProcess* process)
    : ServiceFramework{system, "GsvGpuSession"}, m_process{system.Kernel(), process},
      m_service_context{system, "GsvGpuSession"}, m_worker_client{system.Kernel()} {
    static const FunctionInfo functions[] = {
        {11, &GpuSession::GetEventIdRange, "GetEventIdRange"},
        {12, &GpuSession::CreateEvent, "CreateEvent"},
        {21, &GpuSession::RequestKUserWorkerSession, "RequestKUserWorkerSession"},
        {22, &GpuSession::GetKUserWorkerSession, "GetKUserWorkerSession"},
    };
    RegisterHandlersTipc(functions);
}

GpuSession::~GpuSession() {
    for (auto* event : m_events) {
        m_service_context.CloseEvent(event);
    }
}

Result GpuSession::CreateContext() {
    const auto guard = LockService();
    R_UNLESS(m_phase == Phase::SessionCreated, Kernel::ResultInvalidState);
    m_phase = Phase::ContextCreated;
    R_SUCCEED();
}

Result GpuSession::InitializeContext() {
    const auto guard = LockService();
    R_UNLESS(m_phase == Phase::ContextCreated, Kernel::ResultInvalidState);
    m_phase = Phase::Initialized;
    R_SUCCEED();
}

void GpuSession::GetEventIdRange(HLERequestContext& ctx) {
    if (!HasInputWords(ctx, 0)) {
        Reply(ctx, Kernel::ResultInvalidSize);
        return;
    }
    if (m_phase != Phase::Initialized) {
        Reply(ctx, Kernel::ResultInvalidState);
        return;
    }
    IPC::ResponseBuilder rb{ctx, 4};
    rb.Push(ResultSuccess);
    rb.Push<u32>(0);
    rb.Push<u32>(static_cast<u32>(m_events.size() - 1));
}

void GpuSession::CreateEvent(HLERequestContext& ctx) {
    const auto result = [&]() -> Result {
        R_UNLESS(HasInputWords(ctx, 1), Kernel::ResultInvalidSize);
        R_UNLESS(m_phase == Phase::Initialized, Kernel::ResultInvalidState);
        IPC::RequestParser rp{ctx};
        const auto id = rp.Pop<u32>();
        R_UNLESS(id < m_events.size(), Kernel::ResultInvalidArgument);
        R_UNLESS(m_events[id] == nullptr, ResultAlreadyCreated);
        auto* event = m_service_context.CreateEvent("GpuUserEvent");
        R_UNLESS(event != nullptr, Kernel::ResultOutOfResource);
        m_events[id] = event;
        ctx.AddCopyObject(&event->GetReadableEvent());
        LOG_DEBUG(Service, "Native GPU event created: id={}", id);
        R_SUCCEED();
    }();
    IPC::ResponseBuilder rb{ctx, 2, result.IsSuccess() ? 1U : 0U};
    rb.Push(result);
}

void GpuSession::RequestKUserWorkerSession(HLERequestContext& ctx) {
    const auto result = [&]() -> Result {
        R_UNLESS(HasInputWords(ctx, 1), Kernel::ResultInvalidSize);
        R_UNLESS(m_phase == Phase::Initialized, Kernel::ResultInvalidState);
        IPC::RequestParser rp{ctx};
        const auto id = rp.Pop<u32>();
        R_UNLESS(id < m_events.size() && m_events[id] != nullptr, Kernel::ResultInvalidArgument);
        R_UNLESS(m_worker_client.IsNull(), ResultAlreadyCreated);
        Kernel::KClientSession* client{};
        R_TRY(ctx.CreateInterfaceSession(
            &client, std::make_shared<KUserWorkerSession>(system, m_process.GetPointerUnsafe())));
        m_worker_client.SetObject(client);
        client->Close(system.Kernel());
        // Signal only after the actual worker endpoint has been registered and retained.
        R_TRY(m_events[id]->Signal(system.Kernel()));
        LOG_DEBUG(Service, "Native GPU worker session ready: event_id={}", id);
        R_SUCCEED();
    }();
    Reply(ctx, result);
}

void GpuSession::GetKUserWorkerSession(HLERequestContext& ctx) {
    const auto result = [&]() -> Result {
        R_UNLESS(HasInputWords(ctx, 0), Kernel::ResultInvalidSize);
        R_UNLESS(m_worker_client.IsNotNull(), Kernel::ResultInvalidState);
        ctx.AddCopyObject(m_worker_client.GetPointerUnsafe());
        R_SUCCEED();
    }();
    IPC::ResponseBuilder rb{ctx, 2, result.IsSuccess() ? 1U : 0U};
    rb.Push(result);
}

} // namespace Service::GSV
