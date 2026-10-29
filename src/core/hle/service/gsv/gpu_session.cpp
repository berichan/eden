// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstring>
#include <new>

#include "core/core.h"
#include "core/hle/kernel/k_client_session.h"
#include "core/hle/kernel/k_event.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_readable_event.h"
#include "core/hle/kernel/svc_results.h"
#include "core/hle/service/gsv/gpu_address_space_pool.h"
#include "core/hle/service/gsv/gpu_session.h"
#include "core/hle/service/gsv/graphics_device_table.h"
#include "core/hle/service/ipc_helpers.h"
#include "video_core/gpu.h"
#include "video_core/memory_manager.h"

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
    explicit KUserWorkerSession(Core::System& system, Kernel::KProcess* process,
                                std::shared_ptr<GraphicsDeviceTable> devices,
                                std::shared_ptr<GpuAddressSpacePool> address_spaces)
        : ServiceFramework{system, "GpuWorkerSession"}, m_process{system.Kernel(), process},
          m_graphics_devices{std::move(devices)}, m_address_spaces{std::move(address_spaces)} {}
    ~KUserWorkerSession() override = default;

private:
    Kernel::KScopedAutoObject<Kernel::KProcess> m_process;
    std::shared_ptr<GraphicsDeviceTable> m_graphics_devices;
    std::shared_ptr<GpuAddressSpacePool> m_address_spaces;
};

} // namespace

GpuSession::GpuSession(Core::System& system, Kernel::KProcess* process)
    : ServiceFramework{system, "GsvGpuSession"}, m_process{system.Kernel(), process},
      m_service_context{system, "GsvGpuSession"}, m_worker_client{system.Kernel()},
      m_graphics_devices{std::make_shared<GraphicsDeviceTable>()},
      m_address_spaces{std::make_shared<GpuAddressSpacePool>()} {
    static const FunctionInfo functions[] = {
        {11, &GpuSession::GetEventIdRange, "GetEventIdRange"},
        {12, &GpuSession::CreateEvent, "CreateEvent"},
        {21, &GpuSession::RequestKUserWorkerSession, "RequestKUserWorkerSession"},
        {22, &GpuSession::GetKUserWorkerSession, "GetKUserWorkerSession"},
        {33, &GpuSession::CreateGpuAddressSpace, "CreateGpuAddressSpace"},
        {34, &GpuSession::GetMaxGpuAddressSpaceId, "GetMaxGpuAddressSpaceId"},
        {41, &GpuSession::CreateGraphicsDevice, "CreateGraphicsDevice"},
        {42, &GpuSession::DestroyGraphicsDevice, "DestroyGraphicsDevice"},
        {201, &GpuSession::GetGpuPartitionRange, "GetUserPartitionRange"},
        {202, &GpuSession::GetGpuPartitionRange, "GetGsvPartitionRange"},
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

Result GpuSession::ExistsAddressSpace(u32 handle) {
    const auto guard = LockService();
    R_UNLESS(m_phase == Phase::Initialized, Kernel::ResultInvalidState);
    R_UNLESS(m_address_spaces->Get(handle) != nullptr, GpuAddressSpacePool::InvalidAddressSpace);
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
            &client, std::make_shared<KUserWorkerSession>(system, m_process.GetPointerUnsafe(),
                                                          m_graphics_devices, m_address_spaces)));
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

void GpuSession::CreateGraphicsDevice(HLERequestContext& ctx) {
    u8 id{};
    const auto result = [&]() -> Result {
        R_UNLESS(HasInputWords(ctx, 6), Kernel::ResultInvalidSize);
        R_UNLESS(m_phase == Phase::Initialized, Kernel::ResultInvalidState);
        IPC::RequestParser rp{ctx};
        const auto name = rp.Pop<u64>();
        const auto memory_budget = rp.Pop<u64>();
        const auto debug_label = rp.Pop<u64>();
        R_TRY(m_graphics_devices->Create(&id, name, memory_budget, debug_label));
        LOG_DEBUG(Service, "Native graphics device allocated: id={}", id);
        R_SUCCEED();
    }();
    IPC::ResponseBuilder rb{ctx, 3};
    rb.Push(result);
    rb.Push<u32>(id);
}

void GpuSession::DestroyGraphicsDevice(HLERequestContext& ctx) {
    const auto result = [&]() -> Result {
        R_UNLESS(HasInputWords(ctx, 1), Kernel::ResultInvalidSize);
        R_UNLESS(m_phase == Phase::Initialized, Kernel::ResultInvalidState);
        IPC::RequestParser rp{ctx};
        const auto id = static_cast<u8>(rp.Pop<u32>());
        R_RETURN(m_graphics_devices->Destroy(id));
    }();
    Reply(ctx, result);
}

void GpuSession::CreateGpuAddressSpace(HLERequestContext& ctx) {
    u32 handle{};
    const auto result = [&]() -> Result {
        R_UNLESS(HasInputWords(ctx, 1), Kernel::ResultInvalidSize);
        R_UNLESS(m_phase == Phase::Initialized && system.IsPoweredOn(), Kernel::ResultInvalidState);
        R_UNLESS(m_address_spaces->HasCapacity(), GpuAddressSpacePool::OutOfAddressSpaces);
        IPC::RequestParser rp{ctx};
        const auto flags = rp.Pop<s32>();
        try {
            auto memory_manager =
                std::make_unique<Tegra::MemoryManager>(system, 40, 1ULL << 40, 16, 12);
            system.GPU().InitAddressSpace(*memory_manager);
            R_TRY(m_address_spaces->Add(&handle, std::move(memory_manager), flags));
        } catch (const std::bad_alloc&) {
            return GpuAddressSpacePool::OutOfAddressSpaces;
        }
        LOG_DEBUG(Service, "Native GPU address space created: handle={:#x}", handle);
        R_SUCCEED();
    }();
    IPC::ResponseBuilder rb{ctx, 3};
    rb.Push(result);
    rb.Push(handle);
}

void GpuSession::GetMaxGpuAddressSpaceId(HLERequestContext& ctx) {
    const auto result = [&]() -> Result {
        R_UNLESS(HasInputWords(ctx, 0), Kernel::ResultInvalidSize);
        R_UNLESS(m_phase == Phase::Initialized, Kernel::ResultInvalidState);
        R_SUCCEED();
    }();
    IPC::ResponseBuilder rb{ctx, 3};
    rb.Push(result);
    rb.Push<u32>(result.IsSuccess() ? static_cast<u32>(GpuAddressSpacePool::Capacity - 1) : 0);
}

void GpuSession::GetGpuPartitionRange(HLERequestContext& ctx) {
    u64 first{};
    u64 last{};
    const auto result = [&]() -> Result {
        R_UNLESS(HasInputWords(ctx, 1), Kernel::ResultInvalidSize);
        R_UNLESS(m_phase == Phase::Initialized, Kernel::ResultInvalidState);
        IPC::RequestParser rp{ctx};
        const auto handle = rp.Pop<u32>();
        const auto partition = ctx.GetCommand() == 201 ? GpuAddressSpacePool::Partition::User
                                                       : GpuAddressSpacePool::Partition::Gsv;
        R_TRY(m_address_spaces->QueryPartitionRange(&first, &last, handle, partition));
        LOG_DEBUG(Service, "Native GPU partition queried: handle={:#x}, command={}", handle,
                  ctx.GetCommand());
        R_SUCCEED();
    }();
    IPC::ResponseBuilder rb{ctx, 6};
    rb.Push(result);
    rb.Push(first);
    rb.Push(last);
}

} // namespace Service::GSV
