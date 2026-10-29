// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <memory>
#include <mutex>
#include <optional>
#include "common/container/unordered_map.h"
#include "core/hle/service/nvdrv/core/container.h"

namespace Tegra {
class MemoryManager;
}

namespace Service::Nvidia::NvCore {

class NativeGpuChannel;

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
};

class NativeGpuChannel {
public:
    NativeGpuChannel(SessionId owner, u32 runlist);
    ~NativeGpuChannel();
    NvResult BindAddressSpace(std::shared_ptr<Tegra::MemoryManager> address_space);
    NvResult BindGroup(std::shared_ptr<NativeGpuSchedulingGroup> group, u32 subcontext);
    NvResult UnbindGroup(const std::shared_ptr<NativeGpuSchedulingGroup>& group);
    std::shared_ptr<Tegra::MemoryManager> GetAddressSpace() const;
    std::shared_ptr<NativeGpuSchedulingGroup> GetGroup() const;
    std::optional<u32> GetSubcontext() const;
    u32 GetRunlist() const;
    bool IsOpen() const;

private:
    friend class NativeGpuChannels;
    void Close();

    const SessionId owner;
    const u32 runlist;
    mutable std::mutex mutex;
    bool open{true};
    std::shared_ptr<Tegra::MemoryManager> address_space;
    std::shared_ptr<NativeGpuSchedulingGroup> group;
    u32 subcontext{};
};

// Each native nvdrv context owns this typed descriptor table. Entries only appear after
// runlist allocation; unrelated device descriptors can never be cast to GPU channels.
class NativeGpuChannels {
public:
    ~NativeGpuChannels();
    NvResult Allocate(DeviceFD fd, SessionId owner, u32 requested_runlist);
    std::shared_ptr<NativeGpuChannel> Get(DeviceFD fd, SessionId owner) const;
    void Close(DeviceFD fd, SessionId owner);

private:
    mutable std::mutex mutex;
    ::Common::unordered_map<DeviceFD, std::shared_ptr<NativeGpuChannel>> channels;
};

} // namespace Service::Nvidia::NvCore
