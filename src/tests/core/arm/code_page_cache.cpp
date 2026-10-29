// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstring>
#include <functional>
#include <utility>
#include <catch2/catch_test_macros.hpp>
#include "core/arm/dynarmic/code_page_cache.h"

namespace {

struct CodeMemory {
    u64 base;
    std::array<u32, 2 * Dynarmic::CODE_PAGE_SIZE / sizeof(u32)> words{};
    size_t reads{};
    bool fail_read{};
    std::function<void()> after_read;

    bool IsValidVirtualAddressRange(u64 address, size_t size) const {
        return address >= base && address - base <= sizeof(words) &&
               size <= sizeof(words) - (address - base);
    }

    bool ReadBlock(u64 address, void* destination, size_t size) {
        ++reads;
        if (fail_read || !IsValidVirtualAddressRange(address, size)) {
            return false;
        }
        std::memcpy(destination, reinterpret_cast<const u8*>(words.data()) + (address - base),
                    size);
        if (after_read) {
            std::exchange(after_read, {})();
        }
        return true;
    }
};

} // namespace

TEST_CASE("Instruction fetch sees breakpoint insertion and removal after invalidation",
          "[core][arm]") {
    for (const u64 base : {0x2000ULL, 0x7100000000ULL}) {
        CAPTURE(base);
        CodeMemory memory{base};
        Core::CodePageCache cache;
        constexpr u32 original = 0xd503201f;
        constexpr u32 breakpoint = 0xd4200000;
        memory.words[3] = original;
        REQUIRE(cache.Read(memory, base + 12) == original);
        REQUIRE(memory.reads == 1);
        memory.words[3] = breakpoint;
        REQUIRE(cache.Read(memory, base + 12) == original);
        cache.Invalidate();
        REQUIRE(cache.Read(memory, base + 12) == breakpoint);
        memory.words[3] = original;
        cache.Invalidate();
        REQUIRE(cache.Read(memory, base + 12) == original);
        REQUIRE(memory.reads == 3);
        memory.words[Dynarmic::CODE_PAGE_SIZE / sizeof(u32)] = 0xe1a00000;
        REQUIRE(cache.Read(memory, base + Dynarmic::CODE_PAGE_SIZE) == 0xe1a00000);
        REQUIRE(memory.reads == 4);
        REQUIRE_FALSE(cache.Read(memory, base - 4));
        REQUIRE_FALSE(cache.Read(memory, base + sizeof(memory.words)));
    }
}

TEST_CASE("Instruction page invalidation during a refill is retained for the next fetch",
          "[core][arm]") {
    CodeMemory memory{0x2000};
    Core::CodePageCache cache;
    memory.words[0] = 1;
    memory.after_read = [&] {
        memory.words[0] = 2;
        cache.Invalidate();
    };
    REQUIRE(cache.Read(memory, memory.base) == 1U);
    REQUIRE(cache.Read(memory, memory.base) == 2U);
    REQUIRE(memory.reads == 2);
    memory.fail_read = true;
    cache.Invalidate();
    REQUIRE_FALSE(cache.Read(memory, memory.base));
    memory.fail_read = false;
    memory.words[0] = 3;
    REQUIRE(cache.Read(memory, memory.base) == 3U);
    REQUIRE(memory.reads == 4);
}
