// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <limits>
#include <optional>
#include <dynarmic/interface/code_page.h>
#include "common/common_types.h"

namespace Core {

// A single JIT reader owns the page; other host threads may only invalidate it.
class CodePageCache {
public:
    template <typename Memory>
    std::optional<u32> Read(Memory& memory, u64 address) {
        if (!memory.IsValidVirtualAddressRange(address, sizeof(u32))) {
            return std::nullopt;
        }
        if (m_invalidated.load(std::memory_order_acquire) &&
            m_invalidated.exchange(false, std::memory_order_acq_rel)) {
            m_page_address = InvalidAddress;
        }
        const u64 page_address = address & ~(Dynarmic::CODE_PAGE_SIZE - 1);
        if (m_page_address != page_address) {
            if (!memory.ReadBlock(page_address, &m_page, sizeof(m_page))) {
                m_page_address = InvalidAddress;
                return std::nullopt;
            }
            m_page_address = page_address;
        }
        return m_page.inst[(address & (Dynarmic::CODE_PAGE_SIZE - 1)) / sizeof(u32)];
    }

    void Invalidate() {
        // The debugger can invalidate code from a different host thread than the JIT reader.
        m_invalidated.store(true, std::memory_order_release);
    }

private:
    static constexpr u64 InvalidAddress = std::numeric_limits<u64>::max();
    Dynarmic::CodePage m_page{};
    u64 m_page_address{InvalidAddress};
    std::atomic<bool> m_invalidated{};
};

} // namespace Core
