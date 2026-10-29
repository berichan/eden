// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <memory>
#include <mutex>
#include <optional>
#include "common/common_types.h"
#include "core/hle/result.h"

namespace Tegra {
class MemoryManager;
}

namespace Service::GSV {

class GpuAddressSpacePool {
public:
    enum class Partition {
        User,
        Gsv,
    };
    static constexpr size_t Capacity = 8;
    static constexpr u32 HandleBase = 0x02000000;
    static constexpr Result OutOfAddressSpaces{0x80f3};
    static constexpr Result InvalidArgument{0x100f3};
    static constexpr Result InvalidAddressSpace{0x280f3};

    GpuAddressSpacePool();
    ~GpuAddressSpacePool();

    bool HasCapacity() const;
    Result Add(u32* out_handle, std::unique_ptr<Tegra::MemoryManager> memory_manager, s32 flags);
    Tegra::MemoryManager* Get(u32 handle) const;
    std::optional<s32> GetFlags(u32 handle) const;
    Result QueryPartitionRange(u64* out_first, u64* out_last, u32 handle,
                               Partition partition) const;

private:
    struct AddressSpace {
        std::unique_ptr<Tegra::MemoryManager> memory_manager;
        s32 flags{};
    };
    // Native address spaces remain cached until the last main/worker endpoint releases the pool.
    mutable std::mutex m_mutex;
    std::array<AddressSpace, Capacity> m_address_spaces;
};

} // namespace Service::GSV
