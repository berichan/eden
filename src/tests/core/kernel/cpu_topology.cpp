// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <catch2/catch_test_macros.hpp>
#include "core/hardware_properties.h"
#include "core/hle/kernel/global_scheduler_context.h"
#include "core/hle/kernel/k_affinity_mask.h"
#include "core/hle/kernel/k_capabilities.h"
#include "core/hle/kernel/k_priority_queue.h"
#include "core/hle/kernel/svc_results.h"

namespace {

using Core::Hardware::CpuProfile;

struct QueueThread {
    struct QueueEntry {
        void Initialize() {
            prev = next = nullptr;
        }
        void SetPrev(QueueThread* value) {
            prev = value;
        }
        void SetNext(QueueThread* value) {
            next = value;
        }
        QueueThread* GetPrev() const {
            return prev;
        }
        QueueThread* GetNext() const {
            return next;
        }
        QueueThread* prev{};
        QueueThread* next{};
    };
    QueueEntry& GetPriorityQueueEntry(s32 core) {
        return entries[core];
    }
    const QueueEntry& GetPriorityQueueEntry(s32 core) const {
        return entries[core];
    }
    const Kernel::KAffinityMask& GetAffinityMask() const {
        return affinity;
    }
    s32 GetActiveCore() const {
        return active_core;
    }
    s32 GetPriority() const {
        return priority;
    }
    bool IsDummyThread() const {
        return false;
    }
    std::array<QueueEntry, Core::Hardware::MAX_CPU_CORES> entries{};
    Kernel::KAffinityMask affinity;
    s32 active_core{};
    s32 priority{28};
};

using Queue =
    Kernel::KPriorityQueue<QueueThread, Kernel::KSchedulerPriorityQueue::NumCores,
                           Kernel::Svc::LowestThreadPriority, Kernel::Svc::HighestThreadPriority>;

constexpr u32 ThreadCapability(u32 first, u32 last) {
    return 7 | (63 << 4) | (16 << 10) | (first << 16) | (last << 24);
}

} // namespace

TEST_CASE("CPU capabilities preserve legacy aliases and expose six native cores",
          "[core][topology]") {
    Kernel::KCapabilities caps;
    const std::array six_cores{ThreadCapability(0, 5)};
    REQUIRE(caps.InitializeForUser(six_cores, nullptr, CpuProfile::Nx).IsSuccess());
    REQUIRE(caps.GetCoreMask() == 0x3f);
    REQUIRE(caps.GetPhysicalCoreMask() == 0xf);
    REQUIRE(caps.InitializeForUser(six_cores, nullptr, CpuProfile::Ounce).IsSuccess());
    REQUIRE(caps.GetCoreMask() == 0x3f);
    REQUIRE(caps.GetPhysicalCoreMask() == 0x3f);

    const std::array core_five{ThreadCapability(5, 5)};
    REQUIRE(caps.InitializeForUser(core_five, nullptr, CpuProfile::Ounce).IsSuccess());
    REQUIRE(caps.GetPhysicalCoreMask() == (1ULL << 5));
    REQUIRE(Core::Hardware::VirtualToPhysicalCore(CpuProfile::Ounce, 5) == 5);
    REQUIRE(Core::Hardware::VirtualToPhysicalCore(CpuProfile::Nx, 5) == 0);

    const std::array legacy_alias{ThreadCapability(63, 63)};
    REQUIRE(caps.InitializeForUser(legacy_alias, nullptr, CpuProfile::Nx).IsSuccess());
    REQUIRE(caps.GetPhysicalCoreMask() == (1ULL << 3));
    REQUIRE(caps.InitializeForUser(legacy_alias, nullptr, CpuProfile::Ounce) ==
            Kernel::ResultInvalidCoreId);
    const std::array unavailable{ThreadCapability(0, 6)};
    REQUIRE(caps.InitializeForUser(unavailable, nullptr, CpuProfile::Ounce) ==
            Kernel::ResultInvalidCoreId);
    REQUIRE(caps.InitializeForKip({}, nullptr, CpuProfile::Ounce).IsSuccess());
    REQUIRE(caps.GetCoreMask() == 0x3f);
    REQUIRE(caps.GetPhysicalCoreMask() == 0x3f);
    REQUIRE(caps.InitializeForKip({}, nullptr, CpuProfile::Nx).IsSuccess());
    REQUIRE(caps.GetPhysicalCoreMask() == 0xf);
}

TEST_CASE("Core five queues and migration remain independent of core zero", "[core][topology]") {
    Queue queue;
    QueueThread main, worker;
    main.active_core = 0;
    main.priority = 44;
    main.affinity.SetAffinityMask(1);
    worker.active_core = 5;
    worker.affinity.SetAffinityMask((1ULL << 4) | (1ULL << 5));
    queue.PushBack(&main);
    queue.PushBack(&worker);
    REQUIRE(queue.GetScheduledFront(0) == &main);
    REQUIRE(queue.GetScheduledFront(5) == &worker);
    REQUIRE(queue.GetScheduledFront(4) == nullptr);
    REQUIRE(queue.GetSuggestedFront(4) == &worker);
    REQUIRE(queue.GetSuggestedFront(0) == nullptr);

    worker.active_core = 4;
    queue.ChangeCore(5, &worker);
    REQUIRE(queue.GetScheduledFront(0) == &main);
    REQUIRE(queue.GetScheduledFront(4) == &worker);
    REQUIRE(queue.GetScheduledFront(5) == nullptr);
    REQUIRE(queue.GetSuggestedFront(5) == &worker);

    const auto old_affinity = worker.affinity;
    worker.active_core = 5;
    worker.affinity.SetAffinityMask(1ULL << 5);
    queue.ChangeAffinityMask(4, old_affinity, &worker);
    REQUIRE(queue.GetScheduledFront(0) == &main);
    REQUIRE(queue.GetScheduledFront(5) == &worker);
    REQUIRE(queue.GetSuggestedFront(4) == nullptr);
    queue.Remove(&worker);
    REQUIRE(queue.GetScheduledFront(5) == nullptr);
    REQUIRE(queue.GetScheduledFront(0) == &main);
    queue.Remove(&main);
}
