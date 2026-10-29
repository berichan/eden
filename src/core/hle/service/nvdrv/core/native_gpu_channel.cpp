// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <limits>
#include <new>
#include "common/assert.h"
#include "core/hle/service/nvdrv/core/native_gpu_channel.h"
#include "video_core/memory_manager.h"

namespace Service::Nvidia::NvCore {

NativeGpuSchedulingGroup::NativeGpuSchedulingGroup(SessionId owner_) : owner{owner_} {}

NvResult NativeGpuSchedulingGroup::SetInterleave(u32 level) {
    if (level > 2) {
        return NvResult::BadParameter;
    }
    std::scoped_lock lock{mutex};
    parameters.interleave_level = level;
    return NvResult::Success;
}

NvResult NativeGpuSchedulingGroup::SetL2Promotion(u32 policy) {
    if (policy != 1 && policy != 2 && policy != 4) {
        return NvResult::BadParameter;
    }
    std::scoped_lock lock{mutex};
    parameters.l2_promotion = policy;
    return NvResult::Success;
}

NativeGpuSchedulingGroup::Parameters NativeGpuSchedulingGroup::GetParameters() const {
    std::scoped_lock lock{mutex};
    return parameters;
}

std::optional<u32> NativeGpuSchedulingGroup::GetRunlist() const {
    std::scoped_lock lock{mutex};
    return runlist;
}

u32 NativeGpuSchedulingGroup::GetChannelCount() const {
    std::scoped_lock lock{mutex};
    return channel_count;
}

NvResult NativeGpuSchedulingGroup::Attach(SessionId channel_owner, u32 channel_runlist,
                                          u32 subcontext, Tegra::MemoryManager* address_space) {
    if (channel_owner.id != owner.id) {
        return NvResult::AccessDenied;
    }
    if (subcontext >= MaxSubcontexts || !address_space) {
        return NvResult::BadParameter;
    }
    std::scoped_lock lock{mutex};
    auto& entry = subcontexts[subcontext];
    if ((runlist && *runlist != channel_runlist) ||
        (entry.channel_count && entry.address_space != address_space)) {
        return NvResult::BadParameter;
    }
    if (channel_count == std::numeric_limits<u32>::max()) {
        return NvResult::InsufficientMemory;
    }
    runlist = channel_runlist;
    entry.address_space = address_space;
    ++entry.channel_count;
    ++channel_count;
    return NvResult::Success;
}

void NativeGpuSchedulingGroup::Detach(u32 subcontext) {
    std::scoped_lock lock{mutex};
    auto& entry = subcontexts[subcontext];
    ASSERT(entry.channel_count && channel_count);
    if (--entry.channel_count == 0) {
        entry.address_space = nullptr;
    }
    --channel_count;
}

NativeGpuChannel::NativeGpuChannel(SessionId owner_, u32 runlist_)
    : owner{owner_}, runlist{runlist_} {}

NativeGpuChannel::~NativeGpuChannel() {
    if (group) {
        group->Detach(subcontext);
    }
}

NvResult NativeGpuChannel::BindAddressSpace(std::shared_ptr<Tegra::MemoryManager> new_space) {
    std::scoped_lock lock{mutex};
    if (!open) {
        return NvResult::InvalidState;
    }
    if (!new_space) {
        return NvResult::BadParameter;
    }
    if (address_space) {
        return NvResult::InvalidState;
    }
    address_space = std::move(new_space);
    return NvResult::Success;
}

NvResult NativeGpuChannel::BindGroup(std::shared_ptr<NativeGpuSchedulingGroup> new_group,
                                     u32 new_subcontext) {
    std::scoped_lock lock{mutex};
    if (!open || group || !address_space) {
        return NvResult::InvalidState;
    }
    if (!new_group) {
        return NvResult::BadParameter;
    }
    const auto result = new_group->Attach(owner, runlist, new_subcontext, address_space.get());
    if (result != NvResult::Success) {
        return result;
    }
    group = std::move(new_group);
    subcontext = new_subcontext;
    return NvResult::Success;
}

NvResult NativeGpuChannel::UnbindGroup(const std::shared_ptr<NativeGpuSchedulingGroup>& old_group) {
    std::scoped_lock lock{mutex};
    if (!open) {
        return NvResult::InvalidState;
    }
    if (!group || group != old_group) {
        return NvResult::BadParameter;
    }
    group->Detach(subcontext);
    group.reset();
    subcontext = 0;
    return NvResult::Success;
}

std::shared_ptr<Tegra::MemoryManager> NativeGpuChannel::GetAddressSpace() const {
    std::scoped_lock lock{mutex};
    return address_space;
}

std::shared_ptr<NativeGpuSchedulingGroup> NativeGpuChannel::GetGroup() const {
    std::scoped_lock lock{mutex};
    return group;
}

std::optional<u32> NativeGpuChannel::GetSubcontext() const {
    std::scoped_lock lock{mutex};
    return group ? std::optional{subcontext} : std::nullopt;
}

u32 NativeGpuChannel::GetRunlist() const {
    return runlist;
}

bool NativeGpuChannel::IsOpen() const {
    std::scoped_lock lock{mutex};
    return open;
}

void NativeGpuChannel::Close() {
    std::scoped_lock lock{mutex};
    open = false;
}

NativeGpuChannels::~NativeGpuChannels() {
    for (const auto& [fd, channel] : channels) {
        channel->Close();
    }
}

NvResult NativeGpuChannels::Allocate(DeviceFD fd, SessionId owner, u32 requested_runlist) {
    if (fd < 0) {
        return NvResult::BadParameter;
    }
    std::scoped_lock lock{mutex};
    if (channels.contains(fd)) {
        return NvResult::InvalidState;
    }
    // The native driver falls back to the primary graphics runlist when no engine matches.
    const u32 runlist = requested_runlist == 1 ? 1 : 0;
    try {
        channels.emplace(fd, std::make_shared<NativeGpuChannel>(owner, runlist));
    } catch (const std::bad_alloc&) {
        return NvResult::InsufficientMemory;
    }
    return NvResult::Success;
}

std::shared_ptr<NativeGpuChannel> NativeGpuChannels::Get(DeviceFD fd, SessionId owner) const {
    std::scoped_lock lock{mutex};
    const auto it = channels.find(fd);
    return it != channels.end() && it->second->owner.id == owner.id ? it->second : nullptr;
}

void NativeGpuChannels::Close(DeviceFD fd, SessionId owner) {
    std::scoped_lock lock{mutex};
    const auto it = channels.find(fd);
    if (it != channels.end() && it->second->owner.id == owner.id) {
        it->second->Close();
        channels.erase(it);
    }
}

} // namespace Service::Nvidia::NvCore
