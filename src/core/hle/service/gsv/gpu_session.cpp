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
#include "core/hle/service/gsv/gpu_memory_pool.h"
#include "core/hle/service/gsv/gpu_session.h"
#include "core/hle/service/gsv/graphics_device_table.h"
#include "core/hle/service/gsv/native_image_layout.h"
#include "core/hle/service/ipc_helpers.h"
#include "video_core/gpu.h"
#include "video_core/host1x/host1x.h"
#include "video_core/memory_manager.h"

namespace Service::GSV {

class GpuUserEvents {
public:
    explicit GpuUserEvents(Core::System& system) : m_context{system, "GpuUserEvents"} {}
    ~GpuUserEvents() {
        for (auto* event : events) {
            m_context.CloseEvent(event);
        }
    }
    KernelHelpers::ServiceContext m_context;
    std::mutex mutex;
    std::array<Kernel::KEvent*, 8> events{};
};

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

class KUserWorkerSession final : public ServiceFramework<KUserWorkerSession> {
public:
    explicit KUserWorkerSession(Core::System& system, Kernel::KProcess* process,
                                std::shared_ptr<GraphicsDeviceTable> devices,
                                std::shared_ptr<GpuAddressSpacePool> address_spaces,
                                std::shared_ptr<GpuMemoryPool> memory,
                                std::shared_ptr<GpuUserEvents> events)
        : ServiceFramework{system, "GpuWorkerSession"}, m_process{system.Kernel(), process},
          m_graphics_devices{std::move(devices)}, m_address_spaces{std::move(address_spaces)},
          m_memory{std::move(memory)}, m_events{std::move(events)} {
        static const FunctionInfo functions[] = {
            {101, &KUserWorkerSession::RegisterMemory, "RegisterMemory"},
            {102, &KUserWorkerSession::UnregisterMemory, "UnregisterMemory"},
            {211, &KUserWorkerSession::MapMemory, "NormalMapGpuMemory4K"},
            {213, &KUserWorkerSession::UnmapMemory, "NormalUnmapGpuMemory4K"},
            {214, &KUserWorkerSession::CreateSparseRegion, "CreateSparseRegion"},
            {215, &KUserWorkerSession::DestroySparseRegion, "DestroySparseRegion"},
        };
        RegisterHandlersTipc(functions);
    }
    ~KUserWorkerSession() override = default;

private:
    void RegisterMemory(HLERequestContext& ctx) {
        u32 handle{};
        const auto result = [&]() -> Result {
            R_UNLESS(HasInputWords(ctx, 6), Kernel::ResultInvalidSize);
            IPC::RequestParser rp{ctx};
            const auto address = rp.Pop<u64>();
            const auto last = rp.Pop<u64>();
            const auto flags = static_cast<u8>(rp.Pop<u32>());
            const auto device_id = static_cast<u8>(rp.Pop<u32>());
            R_RETURN(m_memory->Register(&handle, address, last, flags, device_id));
        }();
        IPC::ResponseBuilder rb{ctx, 3};
        rb.Push(result);
        rb.Push(handle);
    }

    void UnregisterMemory(HLERequestContext& ctx) {
        const auto result = [&]() -> Result {
            R_UNLESS(HasInputWords(ctx, 2), Kernel::ResultInvalidSize);
            IPC::RequestParser rp{ctx};
            const auto handle = rp.Pop<u32>();
            const auto event_id = rp.Pop<u32>();
            const std::scoped_lock lock{m_events->mutex};
            R_UNLESS(event_id < m_events->events.size() && m_events->events[event_id] != nullptr,
                     GraphicsDeviceTable::InvalidArgument);
            R_RETURN(m_memory->Unregister(handle, m_events->events[event_id]));
        }();
        Reply(ctx, result);
    }

    void MapMemory(HLERequestContext& ctx) {
        u8 kind{};
        const auto result = [&]() -> Result {
            R_UNLESS(HasInputWords(ctx, 16), Kernel::ResultInvalidSize);
            IPC::RequestParser rp{ctx};
            const auto address_space = rp.Pop<u32>();
            const auto address = rp.Pop<u64>();
            const auto last = rp.Pop<u64>();
            const auto handle = rp.Pop<u32>();
            const auto offset = rp.Pop<u64>();
            GpuMemoryPool::MapOptions options;
            options.permission = static_cast<u8>(rp.Pop<u32>());
            options.cacheability = static_cast<u8>(rp.Pop<u32>());
            options.kind = static_cast<u8>(rp.Pop<u32>());
            options.compressed_kind = static_cast<u8>(rp.Pop<u32>());
            options.raw_texture = static_cast<u8>(rp.Pop<u32>());
            options.device_id = static_cast<u8>(rp.Pop<u32>());
            options.sync = static_cast<u8>(rp.Pop<u32>());
            const auto event_id = rp.Pop<u32>();
            const std::scoped_lock lock{m_events->mutex};
            R_UNLESS(event_id < m_events->events.size() && m_events->events[event_id],
                     GraphicsDeviceTable::InvalidArgument);
            R_TRY(m_memory->Map(&kind, *m_address_spaces, system.Host1x().MemoryManager(),
                                address_space, address, last, handle, offset, options));
            // Completion reports a mapping that is already visible through the canonical GMMU.
            R_RETURN(m_events->events[event_id]->Signal(system.Kernel()));
        }();
        IPC::ResponseBuilder rb{ctx, 3};
        rb.Push(result);
        rb.Push<u32>(kind);
    }

    void UnmapMemory(HLERequestContext& ctx) {
        u64 unmapped{};
        const auto result = [&]() -> Result {
            R_UNLESS(HasInputWords(ctx, 7), Kernel::ResultInvalidSize);
            IPC::RequestParser rp{ctx};
            const auto address_space = rp.Pop<u32>();
            const auto address = rp.Pop<u64>();
            const auto last = rp.Pop<u64>();
            const auto sync = static_cast<u8>(rp.Pop<u32>());
            const auto event_id = rp.Pop<u32>();
            R_UNLESS(sync <= 1, GraphicsDeviceTable::InvalidArgument);
            const std::scoped_lock lock{m_events->mutex};
            R_UNLESS(event_id < m_events->events.size() && m_events->events[event_id],
                     GraphicsDeviceTable::InvalidArgument);
            R_TRY(m_memory->Unmap(address_space, address, last, &unmapped));
            R_RETURN(m_events->events[event_id]->Signal(system.Kernel()));
        }();
        IPC::ResponseBuilder rb{ctx, 4};
        rb.Push(result);
        rb.Push<u64>(unmapped);
    }

    void CreateSparseRegion(HLERequestContext& ctx) {
        const auto result = [&]() -> Result {
            R_UNLESS(HasInputWords(ctx, 10), Kernel::ResultInvalidSize);
            IPC::RequestParser rp{ctx};
            const auto address_space = rp.Pop<u32>();
            const auto address = rp.Pop<u64>();
            const auto last = rp.Pop<u64>();
            const auto page_bits = static_cast<u8>(rp.Pop<u32>());
            const auto alloc_mode = static_cast<u8>(rp.Pop<u32>());
            const auto device_id = static_cast<u8>(rp.Pop<u32>());
            const auto sync = static_cast<u8>(rp.Pop<u32>());
            const auto event_id = rp.Pop<u32>();
            R_UNLESS(sync <= 1, GraphicsDeviceTable::InvalidArgument);
            const std::scoped_lock lock{m_events->mutex};
            R_UNLESS(event_id < m_events->events.size() && m_events->events[event_id],
                     GraphicsDeviceTable::InvalidArgument);
            R_TRY(m_memory->CreateSparseRegion(*m_address_spaces, address_space, address, last,
                                               page_bits, alloc_mode, device_id));
            R_RETURN(m_events->events[event_id]->Signal(system.Kernel()));
        }();
        Reply(ctx, result);
    }

    void DestroySparseRegion(HLERequestContext& ctx) {
        const auto result = [&]() -> Result {
            R_UNLESS(HasInputWords(ctx, 7), Kernel::ResultInvalidSize);
            IPC::RequestParser rp{ctx};
            const auto address_space = rp.Pop<u32>();
            const auto address = rp.Pop<u64>();
            const auto last = rp.Pop<u64>();
            const auto sync = static_cast<u8>(rp.Pop<u32>());
            const auto event_id = rp.Pop<u32>();
            R_UNLESS(sync <= 1, GraphicsDeviceTable::InvalidArgument);
            const std::scoped_lock lock{m_events->mutex};
            R_UNLESS(event_id < m_events->events.size() && m_events->events[event_id],
                     GraphicsDeviceTable::InvalidArgument);
            R_TRY(m_memory->DestroySparseRegion(address_space, address, last));
            R_RETURN(m_events->events[event_id]->Signal(system.Kernel()));
        }();
        Reply(ctx, result);
    }

    Kernel::KScopedAutoObject<Kernel::KProcess> m_process;
    std::shared_ptr<GraphicsDeviceTable> m_graphics_devices;
    std::shared_ptr<GpuAddressSpacePool> m_address_spaces;
    std::shared_ptr<GpuMemoryPool> m_memory;
    std::shared_ptr<GpuUserEvents> m_events;
};

} // namespace

GpuSession::GpuSession(Core::System& system, Kernel::KProcess* process,
                       std::shared_ptr<GpuAddressSpacePool> address_spaces)
    : ServiceFramework{system, "GsvGpuSession"}, m_process{system.Kernel(), process},
      m_events{std::make_shared<GpuUserEvents>(system)}, m_worker_client{system.Kernel()},
      m_graphics_devices{std::make_shared<GraphicsDeviceTable>()},
      m_address_spaces{address_spaces ? std::move(address_spaces)
                                      : std::make_shared<GpuAddressSpacePool>()},
      m_memory{std::make_shared<GpuMemoryPool>(system.Kernel(), process, m_graphics_devices)} {
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

GpuSession::~GpuSession() = default;

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

Result GpuSession::GetNativeMemoryPool(std::shared_ptr<GpuMemoryPool>* out) {
    const auto lock = LockService();
    R_UNLESS(m_phase == Phase::Initialized, Kernel::ResultInvalidState);
    *out = m_memory;
    R_SUCCEED();
}

Result GpuSession::CalculateNativeImageSize(u64* out_size, const NativeImagePreference& preference,
                                            u32 layers) {
    const auto guard = LockService();
    R_UNLESS(m_phase == Phase::Initialized, Kernel::ResultInvalidState);
    NativeImageLayout layout;
    R_TRY(CalculateNativeImageLayout(&layout, preference, layers));
    *out_size = layout.size;
    LOG_DEBUG(Service,
              "Native image size calculated: width={}, height={}, pitch={:#x}, block_height={}, "
              "layers={}, size={:#x}",
              layout.width, layout.height, layout.row_pitch, layout.block_height_log2, layers,
              layout.size);
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
    rb.Push<u32>(static_cast<u32>(m_events->events.size() - 1));
}

void GpuSession::CreateEvent(HLERequestContext& ctx) {
    const auto result = [&]() -> Result {
        R_UNLESS(HasInputWords(ctx, 1), Kernel::ResultInvalidSize);
        R_UNLESS(m_phase == Phase::Initialized, Kernel::ResultInvalidState);
        IPC::RequestParser rp{ctx};
        const auto id = rp.Pop<u32>();
        const std::scoped_lock lock{m_events->mutex};
        R_UNLESS(id < m_events->events.size(), Kernel::ResultInvalidArgument);
        R_UNLESS(m_events->events[id] == nullptr, ResultAlreadyCreated);
        auto* event = m_events->m_context.CreateEvent("GpuUserEvent");
        R_UNLESS(event != nullptr, Kernel::ResultOutOfResource);
        m_events->events[id] = event;
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
        const std::scoped_lock lock{m_events->mutex};
        R_UNLESS(id < m_events->events.size() && m_events->events[id] != nullptr,
                 Kernel::ResultInvalidArgument);
        R_UNLESS(m_worker_client.IsNull(), ResultAlreadyCreated);
        Kernel::KClientSession* client{};
        R_TRY(ctx.CreateInterfaceSession(
            &client, std::make_shared<KUserWorkerSession>(system, m_process.GetPointerUnsafe(),
                                                          m_graphics_devices, m_address_spaces,
                                                          m_memory, m_events)));
        m_worker_client.SetObject(client);
        client->Close(system.Kernel());
        // Signal only after the actual worker endpoint has been registered and retained.
        R_TRY(m_events->events[id]->Signal(system.Kernel()));
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
