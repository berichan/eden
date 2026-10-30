// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cstring>
#include <catch2/catch_test_macros.hpp>
#include "video_core/control/native_constant_buffer.h"

namespace {
using Buffer = Tegra::Control::NativeConstantBuffer;
struct Host {
    std::array<u8, 0x10000> memory{};
    u32 waits{};
    u32 writes{};
    bool writable{true};
    bool accept_wait{true};
    bool accept_write{true};
    bool cancelled{};
    bool cancel_during_wait{};
    Buffer::Host Bind() {
        return {
            this,
            [](void* opaque, GPUVAddr address, size_t size) {
                const auto& self = *static_cast<Host*>(opaque);
                return self.writable && address <= self.memory.size() &&
                       size <= self.memory.size() - address;
            },
            [](void* opaque) {
                auto& self = *static_cast<Host*>(opaque);
                ++self.waits;
                self.cancelled = self.cancel_during_wait;
                return self.accept_wait;
            },
            [](void* opaque, GPUVAddr address, std::span<const u32> words) {
                auto& self = *static_cast<Host*>(opaque);
                ++self.writes;
                if (!self.accept_write || address > self.memory.size() ||
                    words.size_bytes() > self.memory.size() - address) {
                    return false;
                }
                std::memcpy(self.memory.data() + address, words.data(), words.size_bytes());
                return true;
            },
            [](void* opaque) { return static_cast<Host*>(opaque)->cancelled; },
        };
    }
};
} // namespace

TEST_CASE("Native constant-buffer uploads advance only after complete checked writes",
          "[ipc][video_core]") {
    Host host;
    host.memory.fill(0xa5);
    Buffer::Selection selection{32, 0, 0, 4};
    const std::array<u32, 2> words{0x12345678, 0x87654321};
    REQUIRE(Buffer::Upload(selection, words, host.Bind()) == Buffer::Error::None);
    REQUIRE(selection.offset == 12);
    REQUIRE(host.waits == 1);
    REQUIRE(host.writes == 1);
    REQUIRE(std::memcmp(host.memory.data() + 4, words.data(), sizeof(words)) == 0);
    REQUIRE(host.memory[3] == 0xa5);
    REQUIRE(host.memory[12] == 0xa5);
    selection.offset = 28;
    REQUIRE(Buffer::Upload(selection, words, host.Bind()) == Buffer::Error::OutOfBounds);
    REQUIRE(selection.offset == 28);
    REQUIRE(host.waits == 1);
    REQUIRE(host.writes == 1);
    REQUIRE(host.memory[28] == 0xa5);
}

TEST_CASE("Native constant-buffer errors do not write or advance the cursor", "[ipc][video_core]") {
    Host host;
    const std::array<u32, 1> words{0x12345678};
    for (auto selection : std::array<Buffer::Selection, 6>{{{0x20000, 0, 0, 0},
                                                            {32, 0x100, 0, 0},
                                                            {32, 0, 1, 0},
                                                            {32, 0, 0, 2},
                                                            {32, 0, 0, 0x10000},
                                                            {32, 0xff, 0xfffffff0, 0}}}) {
        REQUIRE(Buffer::Upload(selection, words, host.Bind()) != Buffer::Error::None);
    }
    REQUIRE(host.waits == 0);
    REQUIRE(host.writes == 0);
    Buffer::Selection selection{32, 0, 0, 0};
    host.writable = false;
    REQUIRE(Buffer::Upload(selection, words, host.Bind()) == Buffer::Error::MemoryFault);
    REQUIRE(host.waits == 0);
    host.writable = true;
    host.accept_wait = false;
    REQUIRE(Buffer::Upload(selection, words, host.Bind()) == Buffer::Error::WaitFailed);
    REQUIRE(host.writes == 0);
    host.accept_wait = true;
    host.cancel_during_wait = true;
    REQUIRE(Buffer::Upload(selection, words, host.Bind()) == Buffer::Error::Cancelled);
    REQUIRE(host.writes == 0);
    host.cancelled = true;
    REQUIRE(Buffer::Upload(selection, words, host.Bind()) == Buffer::Error::Cancelled);
    host.cancelled = false;
    host.cancel_during_wait = false;
    host.accept_write = false;
    REQUIRE(Buffer::Upload(selection, words, host.Bind()) == Buffer::Error::MemoryFault);
    REQUIRE(host.writes == 1);
    REQUIRE(selection.offset == 0);
    REQUIRE(std::all_of(host.memory.begin(), host.memory.end(), [](u8 byte) { return byte == 0; }));
}

TEST_CASE("Native constant-buffer end-of-window writes never silently wrap", "[ipc][video_core]") {
    Host host;
    Buffer::Selection selection{0x10000, 0, 0, 0xfffc};
    const std::array<u32, 1> words{0x12345678};
    REQUIRE(Buffer::Upload(selection, words, host.Bind()) == Buffer::Error::None);
    REQUIRE(selection.offset == 0x10000);
    REQUIRE(Buffer::Upload(selection, words, host.Bind()) == Buffer::Error::InvalidSelection);
    REQUIRE(host.writes == 1);
    REQUIRE(host.memory[0] == 0);
    selection = {0, 0, 0, 0};
    REQUIRE(Buffer::Upload(selection, words, host.Bind()) == Buffer::Error::OutOfBounds);
    REQUIRE(host.writes == 1);
}
