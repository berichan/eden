// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstring>

#include "core/core.h"
#include "core/hle/kernel/k_event.h"
#include "core/hle/kernel/k_readable_event.h"
#include "core/hle/kernel/svc_results.h"
#include "core/hle/service/gsv/syncpt_session.h"
#include "core/hle/service/ipc_helpers.h"
#include "video_core/host1x/host1x.h"

namespace Service::GSV {
namespace {

constexpr Result ResultInvalidEventCount{0x80f3};
constexpr Result ResultInvalidEvent{0x100f3};
constexpr Result ResultAlreadyInitialized{0x200f3};

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

SyncptSession::SyncptSession(Core::System& system, Tegra::Host1x::SyncpointManager* syncpoints)
    : ServiceFramework{system, "GsvSyncptSession"}, m_service_context{system, "GsvSyncptSession"},
      m_syncpoints{syncpoints} {
    static const FunctionInfo functions[] = {
        {0, &SyncptSession::Ping, "Ping"},
        {1, &SyncptSession::GetMaxEventCount, "GetMaxEventCount"},
        {21, &SyncptSession::SetEventCount, "SetEventCount"},
        {22, &SyncptSession::CreateEvent, "CreateEvent"},
        {103, &SyncptSession::QueryCounter, "QueryCounter"},
    };
    RegisterHandlersTipc(functions);
}

SyncptSession::~SyncptSession() {
    for (auto* event : m_events) {
        m_service_context.CloseEvent(event);
    }
}

void SyncptSession::Ping(HLERequestContext& ctx) {
    if (!HasInputWords(ctx, 2)) {
        Reply(ctx, Kernel::ResultInvalidSize);
        return;
    }
    IPC::RequestParser rp{ctx};
    const auto value = rp.Pop<u64>();
    IPC::ResponseBuilder rb{ctx, 4};
    rb.Push(ResultSuccess);
    rb.Push(value);
    LOG_DEBUG(Service, "GSV syncpoint session ping: value={:#x}", value);
}

void SyncptSession::GetMaxEventCount(HLERequestContext& ctx) {
    if (!HasInputWords(ctx, 0)) {
        Reply(ctx, Kernel::ResultInvalidSize);
        return;
    }
    IPC::ResponseBuilder rb{ctx, 3};
    rb.Push(ResultSuccess);
    rb.Push(MaxEventCount);
}

void SyncptSession::SetEventCount(HLERequestContext& ctx) {
    if (!HasInputWords(ctx, 1)) {
        Reply(ctx, Kernel::ResultInvalidSize);
        return;
    }
    IPC::RequestParser rp{ctx};
    const auto count = rp.Pop<u32>();
    if (m_event_count != 0) {
        Reply(ctx, ResultAlreadyInitialized);
        return;
    }
    if (count > MaxEventCount) {
        Reply(ctx, ResultInvalidEventCount);
        return;
    }
    m_event_count = count + 1;
    LOG_DEBUG(Service, "GSV syncpoint event pool configured: {} client events", count);
    Reply(ctx, ResultSuccess);
}

void SyncptSession::CreateEvent(HLERequestContext& ctx) {
    const auto result = [&]() -> Result {
        R_UNLESS(HasInputWords(ctx, 1), Kernel::ResultInvalidSize);
        IPC::RequestParser rp{ctx};
        const auto index = rp.Pop<u32>();
        R_UNLESS(index < m_event_count, ResultInvalidEvent);
        R_UNLESS(m_events[index] == nullptr, ResultAlreadyInitialized);
        auto* event = m_service_context.CreateEvent("GsvSyncptEvent");
        R_UNLESS(event != nullptr, Kernel::ResultOutOfResource);
        m_events[index] = event;
        ctx.AddCopyObject(&event->GetReadableEvent());
        LOG_DEBUG(Service, "GSV syncpoint event created: index={}", index);
        R_SUCCEED();
    }();
    IPC::ResponseBuilder rb{ctx, 2, result.IsSuccess() ? 1U : 0U};
    rb.Push(result);
}

void SyncptSession::QueryCounter(HLERequestContext& ctx) {
    u64 value{};
    const auto result = [&]() -> Result {
        R_UNLESS(HasInputWords(ctx, 1), Kernel::ResultInvalidSize);
        IPC::CommandHeader header{};
        std::memcpy(&header, ctx.CommandBuffer(), sizeof(header));
        R_UNLESS(header.num_buf_a_descriptors == 0 && header.num_buf_b_descriptors == 0 &&
                     header.num_buf_x_descriptors == 0 && header.num_buf_w_descriptors == 0 &&
                     header.enable_handle_descriptor == 0 &&
                     header.buf_c_descriptor_flags ==
                         IPC::CommandHeader::BufferDescriptorCFlag::Disabled,
                 Kernel::ResultInvalidSize);
        IPC::RequestParser rp{ctx};
        // Native GPU counter identifiers are the bitwise complement of the hardware ID.
        const u32 id = ~rp.Pop<u32>();
        R_UNLESS(id < Tegra::Host1x::SyncpointManager::NativeSyncpointCount,
                 Kernel::ResultNotFound);
        auto& manager = m_syncpoints ? *m_syncpoints : system.Host1x().GetSyncpointManager();
        const auto current = manager.ReadAllocatedHostSyncpoint(id);
        R_UNLESS(current.has_value(), Kernel::ResultNotFound);
        value = *current;
        LOG_DEBUG(Service, "Native syncpoint counter queried: id={}, value={}", id, value);
        R_SUCCEED();
    }();
    IPC::ResponseBuilder rb{ctx, 5};
    rb.Push(result);
    rb.Push(value);
    rb.Push<u32>(0); // Native GPU counters use 32-bit circular comparisons.
}

} // namespace Service::GSV
