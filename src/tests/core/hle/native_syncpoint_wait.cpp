// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <atomic>
#include <latch>
#include <thread>
#include <catch2/catch_test_macros.hpp>
#include "core/hle/kernel/svc_results.h"
#include "core/hle/service/gsv/native_syncpoint_wait_pool.h"
#include "video_core/host1x/syncpoint_manager.h"

namespace {
using Manager = Tegra::Host1x::SyncpointManager;
using Pool = Service::GSV::NativeSyncpointWaitPool;

struct Events {
    std::array<std::atomic<bool>, Pool::Capacity> signaled{};
    std::array<std::atomic<u32>, Pool::Capacity> signals{};
    Pool::Host Host() {
        return {
            this,
            [](void* context, u32 slot) { static_cast<Events*>(context)->signaled[slot] = false; },
            [](void* context, u32 slot) {
                auto& self = *static_cast<Events*>(context);
                ++self.signals[slot];
                self.signaled[slot] = true;
            }};
    }
};
} // namespace

TEST_CASE("Native counter pins prevent reuse after the owner closes", "[core][ipc][gsv]") {
    Manager manager;
    auto owner = manager.AllocateNativeChannelSyncpoint();
    REQUIRE(owner.has_value());
    const auto id = owner->GetId();
    auto first = manager.BorrowSyncpoint(id);
    auto second = manager.BorrowSyncpoint(id);
    REQUIRE(first.has_value());
    REQUIRE(second.has_value());
    REQUIRE_FALSE(manager.BorrowSyncpoint(448));
    REQUIRE_FALSE(manager.BorrowSyncpoint(id + 1));
    owner.reset();
    REQUIRE_FALSE(manager.BorrowSyncpoint(id));
    REQUIRE_FALSE(manager.TryReserveSyncpoint(id));
    auto next = manager.AllocateNativeChannelSyncpoint();
    REQUIRE(next.has_value());
    REQUIRE(next->GetId() != id);
    first.reset();
    REQUIRE_FALSE(manager.TryReserveSyncpoint(id));
    second.reset();
    REQUIRE(manager.TryReserveSyncpoint(id));
    auto pin = manager.BorrowSyncpoint(id);
    manager.ReleaseSyncpoint(id);
    REQUIRE_FALSE(manager.TryReserveSyncpoint(id));
    pin.reset();
    REQUIRE(manager.TryReserveSyncpoint(id));
    manager.ReleaseSyncpoint(id);
    REQUIRE(manager.BorrowSyncpoint(0).has_value());
    // Pins, like channel reservations, may be destroyed after the manager.
    auto temporary = std::make_unique<Manager>();
    auto lease = temporary->AllocateNativeChannelSyncpoint();
    auto surviving_pin = temporary->BorrowSyncpoint(lease->GetId());
    temporary.reset();
    lease.reset();
    surviving_pin.reset();
}

TEST_CASE("Native host notifications compare circular counters and permit callback reentry",
          "[core][ipc][gsv]") {
    REQUIRE(Manager::IsNativeReady(0, 0));
    REQUIRE(Manager::IsNativeReady(0, 0xffffffff));
    REQUIRE_FALSE(Manager::IsNativeReady(0xffffffff, 0));
    REQUIRE(Manager::IsNativeReady(0x7fffffff, 0));
    REQUIRE_FALSE(Manager::IsNativeReady(0x80000000, 0));
    Manager manager;
    u32 called{};
    auto future = manager.RegisterNativeHostAction(17, 2, [&] { ++called; });
    auto now = manager.RegisterNativeHostAction(17, 0, [&] {
        ++called;
        manager.DeregisterNativeHostAction(future);
    });
    REQUIRE(called == 1);
    manager.DeregisterNativeHostAction(now);
    manager.DeregisterNativeHostAction(now);
    REQUIRE(manager.PublishNativeCompletion(17, 2, 2));
    REQUIRE(called == 1);
    manager.DeregisterNativeHostAction(future);
    REQUIRE(manager.PublishNativeCompletion(17, 0xfffffffe, 0xfffffffc));
    bool wrapped{};
    auto wrap = manager.RegisterNativeHostAction(17, 0, [&] { wrapped = true; });
    manager.IncrementHost(17);
    REQUIRE_FALSE(wrapped);
    manager.IncrementHost(17);
    REQUIRE(wrapped);
    manager.DeregisterNativeHostAction(wrap);
}

TEST_CASE("Native event waits use completed host work and clear stale signals on rearm",
          "[core][ipc][gsv]") {
    Manager manager;
    auto owner = manager.AllocateNativeChannelSyncpoint();
    REQUIRE(owner.has_value());
    const auto id = owner->GetId();
    Events events;
    Pool pool{manager, events.Host()};
    pool.Enable(0);
    events.signaled[0] = true;
    REQUIRE(pool.Arm(0, id, 2, 0).IsSuccess());
    REQUIRE_FALSE(events.signaled[0]);
    manager.IncrementGuest(id);
    manager.IncrementGuest(id);
    REQUIRE_FALSE(events.signaled[0]);
    manager.IncrementHost(id);
    REQUIRE_FALSE(events.signaled[0]);
    manager.IncrementHost(id);
    REQUIRE(events.signaled[0]);
    REQUIRE(events.signals[0] == 1);
    events.signaled[0] = false; // The SDK consumed completion before cancellation.
    REQUIRE(pool.Cancel(0).IsSuccess());
    REQUIRE(events.signaled[0]);
    REQUIRE(events.signals[0] == 2);
    REQUIRE(manager.GetHostSyncpointValue(id) == 2);
    REQUIRE(manager.GetGuestSyncpointValue(id) == 2);
    // Mode zero ignores the high word of the 64-bit target, as the SDK does.
    REQUIRE(pool.Arm(0, id, 0xabc00000002ULL, 0).IsSuccess());
    REQUIRE(events.signaled[0]);
    REQUIRE(events.signals[0] == 3);
    REQUIRE(pool.Arm(0, id, 3, 0).IsSuccess());
    REQUIRE_FALSE(events.signaled[0]);
    owner.reset();
    REQUIRE_FALSE(manager.TryReserveSyncpoint(id));
    manager.IncrementHost(id);
    REQUIRE(events.signaled[0]);
    REQUIRE(manager.TryReserveSyncpoint(id));
    manager.ReleaseSyncpoint(id);
}

TEST_CASE("Invalid native wait requests preserve an existing wait", "[core][ipc][gsv]") {
    Manager manager;
    auto owner = manager.AllocateNativeChannelSyncpoint();
    const auto id = owner->GetId();
    Events events;
    Pool pool{manager, events.Host()};
    pool.Enable(0);
    REQUIRE(pool.Arm(0, id, 1, 0).IsSuccess());
    for (const u32 slot : {1U, 64U, 0x10000U, 0xffffffffU}) {
        REQUIRE(pool.Arm(slot, id, 0, 0) == Pool::InvalidEvent);
        REQUIRE(pool.Cancel(slot) == Pool::InvalidEvent);
    }
    REQUIRE(pool.Arm(0, 448, 0, 0) == Kernel::ResultNotFound);
    REQUIRE(pool.Arm(0, id + 1, 0, 0) == Kernel::ResultNotFound);
    REQUIRE(pool.Arm(0, id, 0, 1) == Kernel::ResultNotImplemented);
    REQUIRE(pool.Arm(0, id, 0, 255) == Kernel::ResultNotImplemented);
    REQUIRE_FALSE(events.signaled[0]);
    REQUIRE(manager.PublishNativeCompletion(id, 1, 1));
    REQUIRE(events.signaled[0]);
    REQUIRE(events.signals[0] == 1);
    REQUIRE(pool.Cancel(0).IsSuccess());
    REQUIRE(pool.Cancel(0).IsSuccess());
    REQUIRE(events.signals[0] == 3);
    REQUIRE(manager.GetHostSyncpointValue(id) == 1);
}

TEST_CASE("Cancellation and slot reuse suppress a notification already queued for delivery",
          "[core][ipc][gsv]") {
    Manager manager;
    auto owner = manager.AllocateNativeChannelSyncpoint();
    const auto id = owner->GetId();
    Events events;
    Pool pool{manager, events.Host()};
    pool.Enable(0);
    std::latch entered{1}, proceed{1};
    auto blocker = manager.RegisterNativeHostAction(id, 1, [&] {
        entered.count_down();
        proceed.wait();
    });
    REQUIRE(pool.Arm(0, id, 1, 0).IsSuccess());
    std::jthread producer{[&] { manager.IncrementHost(id); }};
    entered.wait();
    const auto cancel = pool.Cancel(0);
    const auto rearm = pool.Arm(0, id, 2, 0);
    proceed.count_down();
    producer.join();
    REQUIRE(cancel.IsSuccess());
    REQUIRE(rearm.IsSuccess());
    REQUIRE(events.signals[0] == 1); // Only cancellation was acknowledged.
    REQUIRE_FALSE(events.signaled[0]);
    manager.IncrementHost(id);
    REQUIRE(events.signals[0] == 2);
    REQUIRE(events.signaled[0]);
    manager.DeregisterNativeHostAction(blocker);
}

TEST_CASE("Closing a wait pool prevents deferred callbacks from using closed events",
          "[core][ipc][gsv]") {
    Manager manager;
    auto owner = manager.AllocateNativeChannelSyncpoint();
    const auto id = owner->GetId();
    Events events;
    auto pool = std::make_unique<Pool>(manager, events.Host());
    pool->Enable(0);
    std::latch entered{1}, proceed{1};
    auto blocker = manager.RegisterNativeHostAction(id, 1, [&] {
        entered.count_down();
        proceed.wait();
    });
    REQUIRE(pool->Arm(0, id, 1, 0).IsSuccess());
    std::jthread producer{[&] { manager.IncrementHost(id); }};
    entered.wait();
    pool.reset();
    owner.reset();
    const bool reusable = manager.TryReserveSyncpoint(id);
    proceed.count_down();
    producer.join();
    REQUIRE(reusable);
    REQUIRE(events.signals[0] == 0);
    manager.DeregisterNativeHostAction(blocker);
    manager.ReleaseSyncpoint(id);
}
