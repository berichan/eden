// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <bit>
#include <iterator>
#include <limits>
#include <new>
#include <stdexcept>
#include "common/assert.h"
#include "core/hle/service/nvdrv/core/native_gpu_channel.h"
#include "core/hle/service/nvdrv/core/native_gpu_error_notifier.h"
#include "core/hle/service/nvdrv/core/native_gpu_submission.h"
#include "video_core/control/native_gpu_context.h"
#include "video_core/host1x/syncpoint_shim.h"
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

NvResult NativeGpuSchedulingGroup::AcquireGraphicsContext(
    std::shared_ptr<Tegra::Control::NativeGpuGraphicsContext>* out_context, u32 subcontext,
    const std::shared_ptr<Tegra::MemoryManager>& address_space) {
    std::scoped_lock lock{mutex};
    if (subcontext >= MaxSubcontexts || !runlist || *runlist != 0 ||
        !subcontexts[subcontext].channel_count ||
        subcontexts[subcontext].address_space != address_space.get()) {
        return NvResult::BadParameter;
    }
    const auto existing = graphics_contexts.find(address_space.get());
    if (existing != graphics_contexts.end()) {
        *out_context = existing->second;
        return NvResult::Success;
    }
    try {
        auto state = graphics_state ? graphics_state
                                    : std::make_shared<Tegra::Control::AmpereGraphicsState>();
        auto context =
            std::make_shared<Tegra::Control::NativeGpuGraphicsContext>(state, address_space);
        graphics_contexts.emplace(address_space.get(), context);
        graphics_state = std::move(state);
        *out_context = std::move(context);
    } catch (const std::bad_alloc&) {
        return NvResult::InsufficientMemory;
    }
    return NvResult::Success;
}

NativeGpuChannel::NativeGpuChannel(
    SessionId owner_, u32 runlist_,
    std::shared_ptr<Tegra::Host1x::SyncpointManager::Reservation> syncpoint_)
    : owner{owner_}, runlist{runlist_}, syncpoint{std::move(syncpoint_)},
      error_notifier{std::make_shared<NativeGpuErrorNotifier>()} {}

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
    if (!pending_submissions.empty() || backend) {
        return NvResult::Busy;
    }
    group->Detach(subcontext);
    graphics_context.reset();
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

std::shared_ptr<const NativeGpuChannel::CommandRing> NativeGpuChannel::GetCommandRing() const {
    std::scoped_lock lock{mutex};
    return command_ring;
}

std::shared_ptr<const Tegra::Control::NativeGpuGraphicsContext>
NativeGpuChannel::GetGraphicsContext() const {
    std::scoped_lock lock{mutex};
    return graphics_context;
}

const std::shared_ptr<NativeGpuErrorNotifier>& NativeGpuChannel::GetErrorNotifier() const {
    return error_notifier;
}

std::optional<u32> NativeGpuChannel::GetTimeout() const {
    std::scoped_lock lock{mutex};
    return timeout;
}

NvResult NativeGpuChannel::SetTimeout(u32 timeout_) {
    std::scoped_lock lock{mutex};
    if (!open) {
        return NvResult::InvalidState;
    }
    timeout = timeout_;
    return NvResult::Success;
}

NvResult NativeGpuChannel::AllocateObjectContext(u32 class_id, u32 flags) {
    if (class_id != Tegra::Control::AmpereGraphicsState::ClassId) {
        switch (class_id) {
        case 0x902d:
        case 0xa140:
        case 0xc7c0:
        case 0xc76f:
        case 0xc7b5:
            return NvResult::NotSupported;
        default:
            return NvResult::BadParameter;
        }
    }
    // Nonzero flags request preemption modes that need additional execution state.
    if (flags != 0) {
        return NvResult::NotSupported;
    }
    std::scoped_lock lock{mutex};
    if (!open || !address_space || !group) {
        return NvResult::InvalidState;
    }
    if (runlist != 0) {
        return NvResult::BadParameter;
    }
    if (graphics_context) {
        return NvResult::Success;
    }
    return group->AcquireGraphicsContext(&graphics_context, subcontext, address_space);
}

u32 NativeGpuChannel::GetChannelIndex() const {
    return syncpoint->GetId() - Tegra::Host1x::SyncpointManager::NativeChannelSyncpointBase;
}

NvResult NativeGpuChannel::BindZCull(u64 address, u32 mode) {
    if (mode > 3 || (mode == 2 && address == 0)) {
        return NvResult::BadParameter;
    }
    std::scoped_lock lock{mutex};
    if (!open || !address_space || !group) {
        return NvResult::InvalidState;
    }
    if (runlist != 0) {
        return NvResult::BadParameter;
    }
    if (!graphics_context) {
        const auto result =
            group->AcquireGraphicsContext(&graphics_context, subcontext, address_space);
        if (result != NvResult::Success) {
            return result;
        }
    }
    return graphics_context->BindZCull(address, mode) ? NvResult::Success : NvResult::BadParameter;
}

NvResult NativeGpuChannel::SetClientData(u64 data) {
    std::scoped_lock lock{mutex};
    if (!open) {
        return NvResult::InvalidState;
    }
    client_data = data;
    return NvResult::Success;
}

NvResult NativeGpuChannel::GetClientData(u64* out_data) const {
    std::scoped_lock lock{mutex};
    if (!open) {
        return NvResult::InvalidState;
    }
    *out_data = client_data;
    return NvResult::Success;
}

NvResult NativeGpuChannel::QuerySyncpoint(u64* out_address, NvFence* out_fence,
                                          const Tegra::Host1x::SyncpointManager& manager) const {
    std::scoped_lock lock{mutex};
    if (!open || !address_space) {
        return NvResult::InvalidState;
    }
    const auto base = address_space->GetSyncpointShimAddress();
    if (!base) {
        return NvResult::NotInitialized;
    }
    const u32 id = syncpoint->GetId();
    *out_address = static_cast<u64>(*base) + u64{id} * Tegra::Host1x::SyncpointShim::Stride;
    *out_fence = NvFence{static_cast<s32>(id), manager.GetHostSyncpointValue(id)};
    return NvResult::Success;
}

NvResult NativeGpuChannel::AllocateCommandRing(NvFence* out_fence, u32 entries, u32 jobs, u32 flags,
                                               Tegra::Host1x::SyncpointManager& manager) {
    if (entries < 2 || entries >= 0x20000000 || !std::has_single_bit(entries) ||
        jobs > 0x7ffffffe) {
        return NvResult::BadParameter;
    }
    std::scoped_lock lock{mutex};
    if (!open || !address_space || !group) {
        return NvResult::InvalidState;
    }
    if (command_ring) {
        return NvResult::AlreadyAllocated;
    }
    if (jobs == 0) {
        jobs = (entries + 2) / 3;
    }
    try {
        auto ring = std::make_shared<CommandRing>();
        ring->entries.resize(entries);
        ring->jobs.resize(jobs);
        ring->flags = flags & 3;
        ring->syncpoint = syncpoint;
        ring->address_space = address_space;
        ring->group = group;
        command_ring = std::move(ring);
    } catch (const std::bad_alloc&) {
        return NvResult::InsufficientMemory;
    } catch (const std::length_error&) {
        return NvResult::InsufficientMemory;
    }
    reserved_fence = manager.GetHostSyncpointValue(syncpoint->GetId());
    *out_fence = NvFence{static_cast<s32>(syncpoint->GetId()), reserved_fence};
    return NvResult::Success;
}

u32 NativeGpuChannel::GetRunlist() const {
    return runlist;
}

bool NativeGpuChannel::IsOpen() const {
    std::scoped_lock lock{mutex};
    return open;
}

void NativeGpuChannel::Close() {
    {
        std::scoped_lock lock{mutex};
        open = false;
    }
    cancellation.request_stop();
}

NvResult NativeGpuChannel::ReserveSubmission(std::shared_ptr<NativeGpuSubmission>* out_job,
                                             NvFence* out_fence,
                                             const std::shared_ptr<NativeGpuChannel>& self,
                                             std::span<const u64> entries, u32 flags,
                                             u32 increment) {
    if (*out_job) {
        return NvResult::BadParameter;
    }
    if ((flags & ~0x114U) || (!(flags & 0x100U) && increment != 0)) {
        return NvResult::NotSupported;
    }
    // The native bit-8 extension requests ordered completion increments. Keep the
    // supported count bounded; WAIT and the separate bit-1 increment mode stay rejected.
    if (increment > 64) {
        return NvResult::BadParameter;
    }
    if (entries.empty() || entries.size() > Tegra::Control::NativeCommandStream::MaxEntries) {
        return NvResult::BadParameter;
    }
    std::scoped_lock lock{mutex};
    if (!open || faulted || runlist != 0 || !command_ring || !graphics_context ||
        command_ring->group != group || command_ring->address_space != address_space) {
        return NvResult::InvalidState;
    }
    if (entries.size() > command_ring->entries.size() - 1 - ring_used) {
        return NvResult::Busy;
    }
    const auto slot = std::ranges::find_if(command_ring->jobs,
                                           [](const auto& job) { return job.num_entries == 0; });
    if (slot == command_ring->jobs.end()) {
        return NvResult::Busy;
    }
    if (next_submission == 0 || increment > std::numeric_limits<u32>::max() - reserved_fence) {
        return NvResult::OverFlow;
    }
    const NvFence submitted_fence{static_cast<s32>(syncpoint->GetId()), reserved_fence + increment};
    try {
        auto submission = std::shared_ptr<NativeGpuSubmission>(new NativeGpuSubmission(
            self, std::vector<u64>{entries.begin(), entries.end()}, submitted_fence, increment));
        const u32 slot_index = static_cast<u32>(slot - command_ring->jobs.begin());
        const u32 count = static_cast<u32>(entries.size());
        pending_submissions.push_back({next_submission, slot_index, count});
        pending_submissions.back().increments = increment;
        submission->sequence = next_submission++;
        reserved_fence = submitted_fence.value;
        *slot = {ring_put, count, submitted_fence.value};
        const u32 mask = static_cast<u32>(command_ring->entries.size() - 1);
        for (const u64 entry : entries) {
            command_ring->entries[ring_put] = entry;
            ring_put = (ring_put + 1) & mask;
        }
        ring_used += count;
        *out_fence = submitted_fence;
        *out_job = std::move(submission);
    } catch (const std::bad_alloc&) {
        return NvResult::InsufficientMemory;
    }
    return NvResult::Success;
}

void NativeGpuChannel::ReleaseSubmission(u64 sequence, bool completed) {
    bool cancel_channel{};
    {
        std::scoped_lock lock{mutex};
        const auto found =
            std::ranges::find(pending_submissions, sequence, &PendingSubmission::sequence);
        ASSERT(found != pending_submissions.end() && !found->released);
        if (found->increments && !completed) {
            const bool tail = std::all_of(std::next(found), pending_submissions.end(),
                                          [](const auto& job) { return job.released; });
            if (!found->published && tail) {
                reserved_fence -= found->increments;
            } else if (!found->completion_committed) {
                // Never let later completions bridge a cancelled fence and report it ready.
                faulted = true;
                cancel_channel = true;
            }
        }
        found->released = true;
        command_ring->jobs[found->slot] = {};
        // Out-of-order cancellation cannot reuse entries still covered by an older job.
        while (!pending_submissions.empty() && pending_submissions.front().released) {
            ring_used -= pending_submissions.front().entries;
            pending_submissions.pop_front();
        }
    }
    if (cancel_channel) {
        cancellation.request_stop();
    }
}

NativeGpuChannels::NativeGpuChannels(Tegra::Host1x::SyncpointManager& manager_)
    : manager{manager_} {}

NativeGpuChannels::~NativeGpuChannels() {
    for (const auto& [fd, channel] : channels) {
        channel->Close();
    }
}

NvResult NativeGpuChannels::AllocateObjectContext(DeviceFD fd, SessionId owner, u32 class_id,
                                                  u32 flags) {
    const auto channel = Get(fd, owner);
    return channel ? channel->AllocateObjectContext(class_id, flags) : NvResult::BadParameter;
}

NvResult NativeGpuChannels::SetClientData(DeviceFD fd, SessionId owner, u64 data) {
    const auto channel = Get(fd, owner);
    return channel ? channel->SetClientData(data) : NvResult::BadParameter;
}

NvResult NativeGpuChannels::BindZCull(DeviceFD fd, SessionId owner, u64 address, u32 mode) {
    const auto channel = Get(fd, owner);
    return channel ? channel->BindZCull(address, mode) : NvResult::BadParameter;
}

NvResult NativeGpuChannels::SetTimeout(DeviceFD fd, SessionId owner, u32 timeout) {
    const auto channel = Get(fd, owner);
    return channel ? channel->SetTimeout(timeout) : NvResult::BadParameter;
}

NvResult NativeGpuChannels::GetClientData(u64* out_data, DeviceFD fd, SessionId owner) const {
    const auto channel = Get(fd, owner);
    return channel ? channel->GetClientData(out_data) : NvResult::BadParameter;
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
        auto reservation = manager.AllocateNativeChannelSyncpoint();
        if (!reservation) {
            return NvResult::InsufficientMemory;
        }
        auto syncpoint =
            std::make_shared<Tegra::Host1x::SyncpointManager::Reservation>(std::move(*reservation));
        channels.emplace(fd,
                         std::make_shared<NativeGpuChannel>(owner, runlist, std::move(syncpoint)));
    } catch (const std::bad_alloc&) {
        return NvResult::InsufficientMemory;
    }
    return NvResult::Success;
}

NvResult NativeGpuChannels::AllocateCommandRing(NvFence* out_fence, DeviceFD fd, SessionId owner,
                                                u32 entries, u32 jobs, u32 flags) {
    const auto channel = Get(fd, owner);
    return channel ? channel->AllocateCommandRing(out_fence, entries, jobs, flags, manager)
                   : NvResult::BadParameter;
}

std::shared_ptr<NativeGpuChannel> NativeGpuChannels::Get(DeviceFD fd, SessionId owner) const {
    std::scoped_lock lock{mutex};
    const auto it = channels.find(fd);
    return it != channels.end() && it->second->owner.id == owner.id ? it->second : nullptr;
}

NvResult NativeGpuChannels::QuerySyncpoint(u64* out_address, NvFence* out_fence, DeviceFD fd,
                                           SessionId owner) const {
    const auto channel = Get(fd, owner);
    return channel ? channel->QuerySyncpoint(out_address, out_fence, manager)
                   : NvResult::BadParameter;
}

void NativeGpuChannels::Close(DeviceFD fd, SessionId owner) {
    std::scoped_lock lock{mutex};
    const auto it = channels.find(fd);
    if (it != channels.end() && it->second->owner.id == owner.id) {
        it->second->Close();
        channels.erase(it);
    }
}

NvResult NativeGpuChannels::ReserveSubmission(std::shared_ptr<NativeGpuSubmission>* out_job,
                                              NvFence* out_fence, DeviceFD fd, SessionId owner,
                                              std::span<const u64> entries, u32 flags,
                                              u32 increment) {
    const auto channel = Get(fd, owner);
    return channel
               ? channel->ReserveSubmission(out_job, out_fence, channel, entries, flags, increment)
               : NvResult::BadParameter;
}

} // namespace Service::Nvidia::NvCore
