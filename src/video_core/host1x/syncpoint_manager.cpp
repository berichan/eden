// SPDX-FileCopyrightText: Copyright 2025 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <bitset>
#include <utility>
#include "common/assert.h"
#include "video_core/host1x/syncpoint_manager.h"

namespace Tegra {

namespace Host1x {

struct SyncpointManager::Allocations {
    std::mutex mutex;
    std::bitset<NativeSyncpointCount> reserved;
};

SyncpointManager::SyncpointManager() : allocations{std::make_shared<Allocations>()} {}

bool SyncpointManager::IsFixedSyncpoint(u32 id) {
    // The null counter, VIC, display vblanks, NVDEC and NVJPG are not dynamically allocated.
    return id == 0 || id == 12 || id == 26 || id == 27 || id == 54 || id == 55;
}

bool SyncpointManager::TryReserveSyncpoint(u32 id) {
    if (id >= NativeSyncpointCount || IsFixedSyncpoint(id)) {
        return false;
    }
    std::scoped_lock lock{allocations->mutex};
    if (allocations->reserved.test(id)) {
        return false;
    }
    allocations->reserved.set(id);
    return true;
}

void SyncpointManager::ReleaseSyncpoint(u32 id) {
    std::scoped_lock lock{allocations->mutex};
    ASSERT(id < NativeSyncpointCount && !IsFixedSyncpoint(id) && allocations->reserved.test(id));
    allocations->reserved.reset(id);
}

SyncpointManager::Reservation::Reservation(std::shared_ptr<Allocations> allocations_, u32 id_)
    : allocations{std::move(allocations_)}, id{id_} {}

SyncpointManager::Reservation::Reservation(Reservation&& other) noexcept
    : allocations{std::move(other.allocations)}, id{other.id} {}

SyncpointManager::Reservation::~Reservation() {
    if (allocations) {
        std::scoped_lock lock{allocations->mutex};
        ASSERT(allocations->reserved.test(id));
        allocations->reserved.reset(id);
    }
}

u32 SyncpointManager::Reservation::GetId() const {
    return id;
}

std::optional<SyncpointManager::Reservation> SyncpointManager::AllocateNativeChannelSyncpoint() {
    std::scoped_lock lock{allocations->mutex};
    for (u32 id = NativeChannelSyncpointBase; id < NativeSyncpointCount; ++id) {
        if (!IsFixedSyncpoint(id) && !allocations->reserved.test(id)) {
            allocations->reserved.set(id);
            return Reservation{allocations, id};
        }
    }
    return std::nullopt;
}

SyncpointManager::ActionHandle SyncpointManager::RegisterAction(
    std::atomic<u32>& syncpoint, std::list<RegisteredAction>& action_storage, u32 expected_value,
    std::function<void()>&& action) {
    if (syncpoint.load(std::memory_order_acquire) >= expected_value) {
        action();
        return {};
    }

    std::scoped_lock lk(guard);
    if (syncpoint.load(std::memory_order_relaxed) >= expected_value) {
        action();
        return {};
    }
    auto it = action_storage.begin();
    while (it != action_storage.end()) {
        if (it->expected_value >= expected_value) {
            break;
        }
        ++it;
    }
    return action_storage.emplace(it, expected_value, std::move(action));
}

void SyncpointManager::DeregisterAction(std::list<RegisteredAction>& action_storage,
                                        const ActionHandle& handle) {
    std::scoped_lock lk(guard);

    // We want to ensure the iterator still exists prior to erasing it
    // Otherwise, if an invalid iterator was passed in then it could lead to UB
    // It is important to avoid UB in that case since the deregister isn't called from a locked
    // context
    for (auto it = action_storage.begin(); it != action_storage.end(); it++) {
        if (it == handle) {
            action_storage.erase(it);
            return;
        }
    }
}

void SyncpointManager::DeregisterGuestAction(u32 syncpoint_id, const ActionHandle& handle) {
    DeregisterAction(guest_action_storage[syncpoint_id], handle);
}

void SyncpointManager::DeregisterHostAction(u32 syncpoint_id, const ActionHandle& handle) {
    DeregisterAction(host_action_storage[syncpoint_id], handle);
}

void SyncpointManager::IncrementGuest(u32 syncpoint_id) {
    Increment(syncpoints_guest[syncpoint_id], wait_guest_cv, guest_action_storage[syncpoint_id]);
}

void SyncpointManager::IncrementHost(u32 syncpoint_id) {
    Increment(syncpoints_host[syncpoint_id], wait_host_cv, host_action_storage[syncpoint_id]);
}

void SyncpointManager::WaitGuest(u32 syncpoint_id, u32 expected_value) {
    Wait(syncpoints_guest[syncpoint_id], wait_guest_cv, expected_value);
}

void SyncpointManager::WaitHost(u32 syncpoint_id, u32 expected_value) {
    Wait(syncpoints_host[syncpoint_id], wait_host_cv, expected_value);
}

void SyncpointManager::Increment(std::atomic<u32>& syncpoint, std::condition_variable& wait_cv,
                                 std::list<RegisteredAction>& action_storage) {
    auto new_value{syncpoint.fetch_add(1, std::memory_order_acq_rel) + 1};

    std::scoped_lock lk(guard);
    auto it = action_storage.begin();
    while (it != action_storage.end()) {
        if (it->expected_value > new_value) {
            break;
        }
        it->action();
        it = action_storage.erase(it);
    }
    wait_cv.notify_all();
}

void SyncpointManager::Wait(std::atomic<u32>& syncpoint, std::condition_variable& wait_cv,
                            u32 expected_value) {
    const auto pred = [&]() { return syncpoint.load(std::memory_order_acquire) >= expected_value; };
    if (pred()) {
        return;
    }

    std::unique_lock lk(guard);
    wait_cv.wait(lk, pred);
}

} // namespace Host1x

} // namespace Tegra
