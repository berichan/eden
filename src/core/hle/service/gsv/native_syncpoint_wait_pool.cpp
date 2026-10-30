// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <mutex>
#include <new>
#include <utility>
#include "core/hle/kernel/svc_results.h"
#include "core/hle/service/gsv/native_syncpoint_wait_pool.h"
#include "video_core/host1x/syncpoint_manager.h"

namespace Service::GSV {
using Manager = Tegra::Host1x::SyncpointManager;

struct NativeSyncpointWaitPool::Slot {
    Slot(Host host_, u32 index_) : host{host_}, index{index_} {}
    std::mutex mutex;
    const Host host;
    const u32 index;
    bool alive{true};
    bool enabled{};
    bool completed{true};
    u64 generation{};
    Manager::NativeActionHandle action;
    std::unique_ptr<Manager::Reservation> pin;
};

NativeSyncpointWaitPool::NativeSyncpointWaitPool(Manager& manager, Host host) : m_manager{manager} {
    for (u32 index = 0; index < Capacity; ++index)
        m_slots[index] = std::make_shared<Slot>(host, index);
}

NativeSyncpointWaitPool::~NativeSyncpointWaitPool() {
    for (auto& slot : m_slots) {
        Manager::NativeActionHandle action;
        std::unique_ptr<Manager::Reservation> pin;
        {
            std::scoped_lock lock{slot->mutex};
            slot->alive = false;
            ++slot->generation;
            action = std::move(slot->action);
            pin = std::move(slot->pin);
        }
        m_manager.DeregisterNativeHostAction(action);
    }
}

void NativeSyncpointWaitPool::Enable(u32 index) {
    if (index < Capacity) {
        std::scoped_lock lock{m_slots[index]->mutex};
        m_slots[index]->enabled = true;
    }
}

Result NativeSyncpointWaitPool::Arm(u32 index, u32 counter, u64 target, u8 mode) {
    if (index >= Capacity)
        return InvalidEvent;
    auto slot = m_slots[index];
    {
        std::scoped_lock lock{slot->mutex};
        if (!slot->alive || !slot->enabled)
            return InvalidEvent;
    }
    if (mode != 0)
        return Kernel::ResultNotImplemented;
    auto reservation = m_manager.BorrowSyncpoint(counter);
    if (!reservation)
        return Kernel::ResultNotFound;
    try {
        auto pin = std::make_unique<Manager::Reservation>(std::move(*reservation));
        Manager::NativeActionHandle previous;
        u64 generation;
        {
            std::scoped_lock lock{slot->mutex};
            generation = ++slot->generation;
            previous = std::move(slot->action);
            slot->pin.swap(pin);
            slot->completed = false;
            slot->host.clear(slot->host.context, index);
        }
        m_manager.DeregisterNativeHostAction(previous);
        pin.reset();

        Manager::NativeActionHandle action;
        try {
            // Already-completed counters may call back inline. Do not hold a slot
            // lock across registration, and do not publish a token after cancellation.
            action = m_manager.RegisterNativeHostAction(
                counter, static_cast<u32>(target), [weak = std::weak_ptr{slot}, generation] {
                    auto current = weak.lock();
                    if (!current)
                        return;
                    std::unique_ptr<Manager::Reservation> release;
                    {
                        std::scoped_lock lock{current->mutex};
                        if (!current->alive || current->generation != generation)
                            return;
                        current->completed = true;
                        current->action.reset();
                        release = std::move(current->pin);
                        current->host.signal(current->host.context, current->index);
                    }
                });
        } catch (const std::bad_alloc&) {
            std::unique_ptr<Manager::Reservation> release;
            {
                std::scoped_lock lock{slot->mutex};
                if (slot->generation == generation) {
                    ++slot->generation;
                    slot->completed = true;
                    release = std::move(slot->pin);
                }
            }
            return Kernel::ResultOutOfResource;
        }
        {
            std::scoped_lock lock{slot->mutex};
            if (slot->alive && slot->generation == generation && !slot->completed) {
                slot->action = std::move(action);
            }
        }
        m_manager.DeregisterNativeHostAction(action);
        return ResultSuccess;
    } catch (const std::bad_alloc&) {
        return Kernel::ResultOutOfResource;
    }
}

Result NativeSyncpointWaitPool::Cancel(u32 index) {
    if (index >= Capacity)
        return InvalidEvent;
    const auto& slot = m_slots[index];
    Manager::NativeActionHandle action;
    std::unique_ptr<Manager::Reservation> pin;
    {
        std::scoped_lock lock{slot->mutex};
        if (!slot->alive || !slot->enabled)
            return InvalidEvent;
        ++slot->generation;
        slot->completed = true;
        action = std::move(slot->action);
        pin = std::move(slot->pin);
        // The SDK consumes an event after cancel even if completion raced or was
        // consumed already. This acknowledges cancellation; it advances no counter.
        slot->host.signal(slot->host.context, index);
    }
    m_manager.DeregisterNativeHostAction(action);
    return ResultSuccess;
}

} // namespace Service::GSV
