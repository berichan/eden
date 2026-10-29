// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <optional>
#include "common/common_types.h"
#include "core/hle/result.h"

namespace Service::GSV {

class GraphicsDeviceTable {
public:
    static constexpr size_t Capacity = 8;
    static constexpr Result InvalidArgument{0x100f3};
    static constexpr Result OutOfDevices{0x80f3};
    static constexpr Result InvalidDevice{0x280f3};

    struct Device {
        u64 name;
        u64 memory_budget;
        u64 debug_label;
    };

    Result Create(u8* out_id, u64 name, u64 memory_budget, u64 debug_label) {
        R_UNLESS(name != 0, InvalidArgument);
        for (size_t index = 0; index < m_devices.size(); ++index) {
            if (!m_devices[index]) {
                m_devices[index].emplace(name, memory_budget, debug_label);
                *out_id = static_cast<u8>(index);
                R_SUCCEED();
            }
        }
        return OutOfDevices;
    }

    Result Destroy(u8 id) {
        R_UNLESS(id < m_devices.size(), InvalidArgument);
        R_UNLESS(m_devices[id], InvalidDevice);
        m_devices[id].reset();
        R_SUCCEED();
    }

    const Device* Get(u8 id) const {
        return id < m_devices.size() && m_devices[id] ? &*m_devices[id] : nullptr;
    }

private:
    std::array<std::optional<Device>, Capacity> m_devices;
};

} // namespace Service::GSV
