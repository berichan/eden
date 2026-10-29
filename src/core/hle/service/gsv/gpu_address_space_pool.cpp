// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "core/hle/service/gsv/gpu_address_space_pool.h"
#include "video_core/memory_manager.h"

namespace Service::GSV {

GpuAddressSpacePool::GpuAddressSpacePool() = default;
GpuAddressSpacePool::~GpuAddressSpacePool() = default;

bool GpuAddressSpacePool::HasCapacity() const {
    std::scoped_lock lock{m_mutex};
    for (const auto& address_space : m_address_spaces) {
        if (!address_space.memory_manager) {
            return true;
        }
    }
    return false;
}

Result GpuAddressSpacePool::Add(u32* out_handle,
                                std::unique_ptr<Tegra::MemoryManager> memory_manager, s32 flags) {
    R_UNLESS(memory_manager, InvalidArgument);
    std::scoped_lock lock{m_mutex};
    for (size_t index = 0; index < m_address_spaces.size(); ++index) {
        if (!m_address_spaces[index].memory_manager) {
            m_address_spaces[index] = AddressSpace{std::move(memory_manager), flags};
            *out_handle = HandleBase | static_cast<u32>(index);
            R_SUCCEED();
        }
    }
    return OutOfAddressSpaces;
}

Tegra::MemoryManager* GpuAddressSpacePool::Get(u32 handle) const {
    if (handle < HandleBase || handle - HandleBase >= m_address_spaces.size()) {
        return nullptr;
    }
    std::scoped_lock lock{m_mutex};
    return m_address_spaces[handle - HandleBase].memory_manager.get();
}

std::optional<s32> GpuAddressSpacePool::GetFlags(u32 handle) const {
    if (handle < HandleBase || handle - HandleBase >= m_address_spaces.size()) {
        return std::nullopt;
    }
    std::scoped_lock lock{m_mutex};
    const auto& address_space = m_address_spaces[handle - HandleBase];
    return address_space.memory_manager ? std::optional{address_space.flags} : std::nullopt;
}

Result GpuAddressSpacePool::QueryPartitionRange(u64* out_first, u64* out_last, u32 handle,
                                                Partition partition) const {
    const auto flags = GetFlags(handle);
    R_UNLESS(flags, InvalidAddressSpace);
    if (partition == Partition::User) {
        *out_first = 0x8000000;
        *out_last = 0xbfffffffff;
    } else if (*flags >= 0) {
        *out_first = 0xc800000000;
        *out_last = 0xc8ffffffff;
    } else {
        *out_first = 0xc900000000;
        *out_last = 0xcfffffffff;
    }
    R_SUCCEED();
}

} // namespace Service::GSV
