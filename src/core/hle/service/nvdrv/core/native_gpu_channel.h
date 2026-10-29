// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>
#include "common/container/unordered_map.h"
#include "core/hle/service/nvdrv/core/container.h"
#include "video_core/host1x/syncpoint_manager.h"

namespace Tegra {
class MemoryManager;
namespace Control {
struct AmpereGraphicsState;
class NativeGpuGraphicsContext;
} // namespace Control
} // namespace Tegra

namespace Service::Nvidia::NvCore {

class NativeGpuChannel;
class NativeGpuErrorNotifier;

class NativeGpuSchedulingGroup {
public:
    static constexpr u32 MaxSubcontexts = 64;
    struct Parameters {
        u32 interleave_level{};
        u32 l2_promotion{};
    };

    explicit NativeGpuSchedulingGroup(SessionId owner);
    NvResult SetInterleave(u32 level);
    NvResult SetL2Promotion(u32 policy);
    Parameters GetParameters() const;
    std::optional<u32> GetRunlist() const;
    u32 GetChannelCount() const;

private:
    friend class NativeGpuChannel;
    NvResult Attach(SessionId channel_owner, u32 runlist, u32 subcontext,
                    Tegra::MemoryManager* address_space);
    void Detach(u32 subcontext);
    NvResult AcquireGraphicsContext(
        std::shared_ptr<Tegra::Control::NativeGpuGraphicsContext>* out_context, u32 subcontext,
        const std::shared_ptr<Tegra::MemoryManager>& address_space);

    const SessionId owner;
    mutable std::mutex mutex;
    Parameters parameters;
    std::optional<u32> runlist;
    u32 channel_count{};
    struct Subcontext {
        Tegra::MemoryManager* address_space{};
        u32 channel_count{};
    };
    std::array<Subcontext, MaxSubcontexts> subcontexts{};
    std::shared_ptr<Tegra::Control::AmpereGraphicsState> graphics_state;
    ::Common::unordered_map<Tegra::MemoryManager*,
                            std::shared_ptr<Tegra::Control::NativeGpuGraphicsContext>>
        graphics_contexts;
};

class NativeGpuChannel {
public:
    struct CommandRing {
        struct Job {
            u32 first_entry{};
            u32 num_entries{};
            u32 fence_value{};
        };
        std::vector<u64> entries;
        std::vector<Job> jobs;
        std::shared_ptr<Tegra::Host1x::SyncpointManager::Reservation> syncpoint;
        std::shared_ptr<Tegra::MemoryManager> address_space;
        std::shared_ptr<NativeGpuSchedulingGroup> group;
        u32 flags{};
    };

    NativeGpuChannel(SessionId owner, u32 runlist,
                     std::shared_ptr<Tegra::Host1x::SyncpointManager::Reservation> syncpoint);
    ~NativeGpuChannel();
    NvResult BindAddressSpace(std::shared_ptr<Tegra::MemoryManager> address_space);
    NvResult BindGroup(std::shared_ptr<NativeGpuSchedulingGroup> group, u32 subcontext);
    NvResult UnbindGroup(const std::shared_ptr<NativeGpuSchedulingGroup>& group);
    std::shared_ptr<Tegra::MemoryManager> GetAddressSpace() const;
    std::shared_ptr<NativeGpuSchedulingGroup> GetGroup() const;
    std::optional<u32> GetSubcontext() const;
    std::shared_ptr<const CommandRing> GetCommandRing() const;
    std::shared_ptr<const Tegra::Control::NativeGpuGraphicsContext> GetGraphicsContext() const;
    const std::shared_ptr<NativeGpuErrorNotifier>& GetErrorNotifier() const;
    std::optional<u32> GetTimeout() const;
    u32 GetChannelIndex() const;
    u32 GetRunlist() const;
    bool IsOpen() const;

private:
    friend class NativeGpuChannels;
    void Close();
    NvResult AllocateObjectContext(u32 class_id, u32 flags);
    NvResult SetClientData(u64 data);
    NvResult SetTimeout(u32 timeout);
    NvResult GetClientData(u64* out_data) const;
    NvResult AllocateCommandRing(NvFence* out_fence, u32 entries, u32 jobs, u32 flags,
                                 Tegra::Host1x::SyncpointManager& manager);
    NvResult QuerySyncpoint(u64* out_address, NvFence* out_fence,
                            const Tegra::Host1x::SyncpointManager& manager) const;

    const SessionId owner;
    const u32 runlist;
    const std::shared_ptr<Tegra::Host1x::SyncpointManager::Reservation> syncpoint;
    const std::shared_ptr<NativeGpuErrorNotifier> error_notifier;
    mutable std::mutex mutex;
    bool open{true};
    std::shared_ptr<Tegra::MemoryManager> address_space;
    std::shared_ptr<NativeGpuSchedulingGroup> group;
    u32 subcontext{};
    std::shared_ptr<CommandRing> command_ring;
    std::shared_ptr<Tegra::Control::NativeGpuGraphicsContext> graphics_context;
    u64 client_data{};
    std::optional<u32> timeout;
};

// Each native nvdrv context owns this typed descriptor table. Entries only appear after
// runlist allocation; unrelated device descriptors can never be cast to GPU channels.
class NativeGpuChannels {
public:
    explicit NativeGpuChannels(Tegra::Host1x::SyncpointManager& manager);
    ~NativeGpuChannels();
    NvResult Allocate(DeviceFD fd, SessionId owner, u32 requested_runlist);
    NvResult AllocateObjectContext(DeviceFD fd, SessionId owner, u32 class_id, u32 flags);
    NvResult SetClientData(DeviceFD fd, SessionId owner, u64 data);
    NvResult SetTimeout(DeviceFD fd, SessionId owner, u32 timeout);
    NvResult GetClientData(u64* out_data, DeviceFD fd, SessionId owner) const;
    std::shared_ptr<NativeGpuChannel> Get(DeviceFD fd, SessionId owner) const;
    NvResult AllocateCommandRing(NvFence* out_fence, DeviceFD fd, SessionId owner, u32 entries,
                                 u32 jobs, u32 flags);
    NvResult QuerySyncpoint(u64* out_address, NvFence* out_fence, DeviceFD fd,
                            SessionId owner) const;
    void Close(DeviceFD fd, SessionId owner);

private:
    mutable std::mutex mutex;
    Tegra::Host1x::SyncpointManager& manager;
    ::Common::unordered_map<DeviceFD, std::shared_ptr<NativeGpuChannel>> channels;
};

} // namespace Service::Nvidia::NvCore
