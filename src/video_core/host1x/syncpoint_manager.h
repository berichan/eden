// SPDX-FileCopyrightText: 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <optional>

#include "common/common_types.h"

namespace Tegra {

namespace Host1x {

class SyncpointManager {
    struct Allocations;

public:
    static constexpr size_t NativeSyncpointCount = 448;
    static constexpr u32 NativeChannelSyncpointBase = 17;

    class Reservation {
    public:
        Reservation(Reservation&& other) noexcept;
        Reservation& operator=(Reservation&&) = delete;
        ~Reservation();
        u32 GetId() const;

    private:
        friend class SyncpointManager;
        Reservation(std::shared_ptr<Allocations> allocations, u32 id, bool borrowed = false);
        std::shared_ptr<Allocations> allocations;
        u32 id;
        bool borrowed;
    };

    SyncpointManager();
    static bool IsFixedSyncpoint(u32 id);
    bool TryReserveSyncpoint(u32 id);
    void ReleaseSyncpoint(u32 id);
    std::optional<Reservation> AllocateNativeChannelSyncpoint();
    // Pin the current allocation without extending its owner's lifetime. Retired IDs
    // cannot be reused until all pending waits have released their pins.
    std::optional<Reservation> BorrowSyncpoint(u32 id);
    u32 GetGuestSyncpointValue(u32 id) const {
        return syncpoints_guest[id].load(std::memory_order_acquire);
    }

    u32 GetHostSyncpointValue(u32 id) const {
        return syncpoints_host[id].load(std::memory_order_acquire);
    }
    std::optional<u32> ReadAllocatedHostSyncpoint(u32 id) const;

    struct RegisteredAction {
        explicit RegisteredAction(u32 expected_value_, std::function<void()>&& action_)
            : expected_value{expected_value_}, action{std::move(action_)} {}
        u32 expected_value;
        std::function<void()> action;
    };
    using ActionHandle = std::list<RegisteredAction>::iterator;

    class NativeAction;
    using NativeActionHandle = std::shared_ptr<NativeAction>;
    static bool IsNativeReady(u32 current, u32 target) {
        return u32(current - target) < 0x80000000U;
    }
    // Unlike legacy action iterators, these identities remain safe to cancel after
    // completion or slot reuse. Callbacks run outside the counter lock. The caller
    // retains an allocation pin until cancellation or callback completion.
    NativeActionHandle RegisterNativeHostAction(u32 id, u32 target, std::function<void()> action);
    void DeregisterNativeHostAction(const NativeActionHandle& action);

    template <typename Func>
    ActionHandle RegisterGuestAction(u32 syncpoint_id, u32 expected_value, Func&& action) {
        return RegisterAction(syncpoints_guest[syncpoint_id], guest_action_storage[syncpoint_id],
                              expected_value, std::move(action));
    }

    template <typename Func>
    ActionHandle RegisterHostAction(u32 syncpoint_id, u32 expected_value, Func&& action) {
        return RegisterAction(syncpoints_host[syncpoint_id], host_action_storage[syncpoint_id],
                              expected_value, std::move(action));
    }

    void DeregisterGuestAction(u32 syncpoint_id, const ActionHandle& handle);

    void DeregisterHostAction(u32 syncpoint_id, const ActionHandle& handle);

    void IncrementGuest(u32 syncpoint_id);

    void IncrementHost(u32 syncpoint_id);

    // Commit a bounded native FIFO completion after real work has drained. Both
    // counters become visible before any guest or host notification is invoked.
    bool PublishNativeCompletion(u32 syncpoint_id, u32 target, u32 increments);

    void WaitGuest(u32 syncpoint_id, u32 expected_value);

    void WaitHost(u32 syncpoint_id, u32 expected_value);

    bool IsReadyGuest(u32 syncpoint_id, u32 expected_value) const {
        return syncpoints_guest[syncpoint_id].load(std::memory_order_acquire) >= expected_value;
    }

    bool IsReadyHost(u32 syncpoint_id, u32 expected_value) const {
        return syncpoints_host[syncpoint_id].load(std::memory_order_acquire) >= expected_value;
    }

private:
    struct NativeRegistration {
        u32 target;
        NativeActionHandle action;
    };
    using NativeActions = std::list<NativeRegistration>;
    void CollectNativeActions(u32 id, NativeActions& ready);
    static void RunNativeActions(NativeActions& ready);
    std::shared_ptr<Allocations> allocations;
    void Increment(std::atomic<u32>& syncpoint, std::condition_variable& wait_cv,
                   std::list<RegisteredAction>& action_storage);

    ActionHandle RegisterAction(std::atomic<u32>& syncpoint,
                                std::list<RegisteredAction>& action_storage, u32 expected_value,
                                std::function<void()>&& action);

    void DeregisterAction(std::list<RegisteredAction>& action_storage, const ActionHandle& handle);

    void Wait(std::atomic<u32>& syncpoint, std::condition_variable& wait_cv, u32 expected_value);

    static constexpr size_t NUM_MAX_SYNCPOINTS = NativeSyncpointCount;

    std::array<std::atomic<u32>, NUM_MAX_SYNCPOINTS> syncpoints_guest{};
    std::array<std::atomic<u32>, NUM_MAX_SYNCPOINTS> syncpoints_host{};

    std::array<std::list<RegisteredAction>, NUM_MAX_SYNCPOINTS> guest_action_storage;
    std::array<std::list<RegisteredAction>, NUM_MAX_SYNCPOINTS> host_action_storage;
    std::array<NativeActions, NUM_MAX_SYNCPOINTS> native_host_actions;

    std::mutex guard;
    std::condition_variable wait_guest_cv;
    std::condition_variable wait_host_cv;
};

} // namespace Host1x

} // namespace Tegra
