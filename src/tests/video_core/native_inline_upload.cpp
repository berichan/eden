// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cstring>
#include <catch2/catch_test_macros.hpp>
#include "video_core/control/native_inline_upload.h"

namespace {
using Upload = Tegra::Control::NativeInlineUpload;
struct Host {
    std::array<u8, 512> memory{};
    u32 waits{};
    u32 writes{};
    u32 fail_wait{};
    bool writable{true};
    bool accept_write{true};
    bool cancelled{};
    bool cancel_on_wait{};
    Upload::Host Bind() {
        return {
            this,
            [](void* opaque, GPUVAddr address, size_t bytes) {
                const auto& self = *static_cast<Host*>(opaque);
                return self.writable && address <= self.memory.size() &&
                       bytes <= self.memory.size() - address;
            },
            [](void* opaque) {
                auto& self = *static_cast<Host*>(opaque);
                self.cancelled = self.cancel_on_wait;
                return ++self.waits != self.fail_wait;
            },
            [](void* opaque, GPUVAddr address, std::span<const u8> bytes) {
                auto& self = *static_cast<Host*>(opaque);
                ++self.writes;
                if (!self.accept_write || address > self.memory.size() ||
                    bytes.size() > self.memory.size() - address) {
                    return false;
                }
                std::memcpy(self.memory.data() + address, bytes.data(), bytes.size());
                return true;
            },
            [](void* opaque) { return static_cast<Host*>(opaque)->cancelled; },
        };
    }
};
} // namespace

TEST_CASE("Native inline DMA collects every word before exposing the complete upload",
          "[ipc][video_core]") {
    Host host;
    host.memory.fill(0xa5);
    Upload upload;
    Upload::Parameters parameters{256, 1, 0, 16, 0};
    REQUIRE(upload.Launch(parameters, 0x1011, host.Bind()) == Upload::Error::None);
    parameters.address_lower = 300;
    for (u32 index = 0; index < 63; ++index) {
        REQUIRE(upload.PushWord(0, host.Bind()) == Upload::Error::None);
        REQUIRE(upload.IsPending());
        REQUIRE(host.writes == 0);
        REQUIRE(host.waits == 0);
    }
    REQUIRE(upload.PushWord(0, host.Bind()) == Upload::Error::None);
    REQUIRE_FALSE(upload.IsPending());
    REQUIRE(host.waits == 2);
    REQUIRE(host.writes == 1);
    REQUIRE(host.memory[15] == 0xa5);
    REQUIRE(host.memory[272] == 0xa5);
    REQUIRE(std::all_of(host.memory.begin() + 16, host.memory.begin() + 272,
                        [](u8 byte) { return byte == 0; }));
    REQUIRE(upload.PushWord(0xffffffff, host.Bind()) == Upload::Error::UnexpectedData);
    REQUIRE(host.writes == 1);
}

TEST_CASE("Native inline DMA supports a truncated final word without copying padding",
          "[ipc][video_core]") {
    for (u32 flags : {1U, 0x11U, 0x41U, 0x51U, 0x1001U, 0x1011U, 0x1041U, 0x1051U}) {
        Host host;
        host.memory.fill(0xa5);
        Upload upload;
        REQUIRE(upload.Launch({5, 1, 0, 1, 0}, flags, host.Bind()) == Upload::Error::None);
        REQUIRE(upload.PushWord(0x04030201, host.Bind()) == Upload::Error::None);
        REQUIRE(host.writes == 0);
        REQUIRE(upload.PushWord(0xffeedd05, host.Bind()) == Upload::Error::None);
        const std::array<u8, 7> expected{0xa5, 1, 2, 3, 4, 5, 0xa5};
        REQUIRE(std::equal(expected.begin(), expected.end(), host.memory.begin()));
        REQUIRE(host.waits == ((flags & 0x10) ? 2U : 1U));
    }
}

TEST_CASE("Native inline DMA rejects unsafe launch shapes and overlapping transfers",
          "[ipc][video_core]") {
    Host host;
    for (u32 flags : {0U, 0x21U, 0x1013U, 0x1111U, 0x80001011U}) {
        Upload upload;
        REQUIRE(upload.Launch({4, 1, 0, 0, 0}, flags, host.Bind()) ==
                Upload::Error::UnsupportedLaunch);
        REQUIRE_FALSE(upload.IsPending());
    }
    for (auto parameters : std::array<Upload::Parameters, 5>{{{0, 1, 0, 0, 0},
                                                              {4, 2, 0, 0, 4},
                                                              {Upload::MaxBytes + 1, 1, 0, 0, 0},
                                                              {4, 1, 0x100, 0, 0},
                                                              {4, 1, 0xff, 0xffffffff, 0}}}) {
        Upload upload;
        REQUIRE(upload.Launch(parameters, 0x1011, host.Bind()) == Upload::Error::InvalidParameters);
    }
    Upload upload;
    REQUIRE(upload.Launch({8, 1, 0, 0, 0}, 0x1011, host.Bind()) == Upload::Error::None);
    REQUIRE(upload.PushWord(0, host.Bind()) == Upload::Error::None);
    REQUIRE(upload.Launch({4, 1, 0, 4, 0}, 0x1011, host.Bind()) == Upload::Error::TransferPending);
    REQUIRE(upload.PushWord(0, host.Bind()) == Upload::Error::TransferPending);
    REQUIRE(upload.IsPending());
    REQUIRE(host.waits == 0);
    REQUIRE(host.writes == 0);
}

TEST_CASE("Native inline DMA failures cannot report a completed transfer", "[ipc][video_core]") {
    for (u32 failure = 0; failure < 6; ++failure) {
        Host host;
        Upload upload;
        REQUIRE(upload.Launch({4, 1, 0, 0, 0}, 0x1011, host.Bind()) == Upload::Error::None);
        switch (failure) {
        case 0:
            host.writable = false;
            break;
        case 1:
            host.fail_wait = 1;
            break;
        case 2:
            host.cancel_on_wait = true;
            break;
        case 3:
            host.accept_write = false;
            break;
        case 4:
            host.fail_wait = 2;
            break;
        case 5:
            host.cancelled = true;
            break;
        }
        REQUIRE(upload.PushWord(0x12345678, host.Bind()) != Upload::Error::None);
        REQUIRE(upload.GetError() != Upload::Error::None);
        REQUIRE(upload.IsPending());
        REQUIRE(upload.PushWord(0, host.Bind()) == upload.GetError());
        REQUIRE(host.writes == (failure == 3 || failure == 4 ? 1U : 0U));
    }
    Host host;
    host.writable = false;
    Upload upload;
    REQUIRE(upload.Launch({4, 1, 0, 0, 0}, 0x1011, host.Bind()) == Upload::Error::MemoryFault);
    REQUIRE(host.waits == 0);
    REQUIRE(host.writes == 0);
}
