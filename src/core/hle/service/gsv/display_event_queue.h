// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cstring>
#include <span>
#include "common/common_types.h"
#include "core/hle/result.h"

namespace Service::GSV {

class DisplayEventQueue {
public:
    static constexpr u32 MaxEventId = 1023;
    static constexpr size_t MaxBytes = (MaxEventId + 1) / 8;

    Result Initialize(u32 max_event_id) {
        if (m_word_count != 0) {
            return Result{ErrorModule::VI, 5};
        }
        if (max_event_id > MaxEventId || (max_event_id & 31U) != 31U) {
            return Result{ErrorModule::VI, 4};
        }
        m_word_count = max_event_id / 32 + 1;
        return ResultSuccess;
    }

    bool Notify(u32 id) {
        // Slot zero is reserved; an out-of-pool notification is ignored by the server.
        if (id == 0 || id / 32 >= m_word_count) {
            return false;
        }
        m_pending[id / 32] = m_pending[id / 32] | (1U << (id % 32));
        return true;
    }

    void Drain(std::span<u8> output) {
        const auto size = (std::min)(output.size(), m_word_count * sizeof(u32));
        if (size != 0) {
            std::memcpy(output.data(), m_pending.data(), size);
        }
        std::fill(output.begin() + size, output.end(), u8{0xff});
        m_pending.fill(0);
    }

private:
    std::array<u32_le, (MaxEventId + 1) / 32> m_pending{};
    size_t m_word_count{};
};

} // namespace Service::GSV
