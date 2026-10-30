// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include "video_core/control/native_semaphore_release.h"

namespace {
using Release = Tegra::Control::NativeSemaphoreRelease;
struct Host {
    GPUVAddr address{};
    u32 waits{}, clocks{}, writes{}, payload{};
    u64 timestamp{};
    bool writable{true}, wait_ok{true}, write_ok{true}, stopped{}, cancel_in_wait{},
        cancel_in_write{};
    Release::Host Bind() {
        return {this,
                [](void* opaque, GPUVAddr address, size_t size) {
                    auto& self = *static_cast<Host*>(opaque);
                    self.address = address;
                    REQUIRE(size == 16);
                    return self.writable;
                },
                [](void* opaque) {
                    auto& self = *static_cast<Host*>(opaque);
                    ++self.waits;
                    if (self.cancel_in_wait)
                        self.stopped = true;
                    return self.wait_ok;
                },
                [](void* opaque) -> u64 {
                    auto& self = *static_cast<Host*>(opaque);
                    REQUIRE(self.waits == 1);
                    ++self.clocks;
                    return 0x1122334455667788ULL;
                },
                [](void* opaque, GPUVAddr address, u32 payload, u64 timestamp) {
                    auto& self = *static_cast<Host*>(opaque);
                    REQUIRE(address == self.address);
                    REQUIRE(self.clocks == 1);
                    if (self.cancel_in_write)
                        self.stopped = true;
                    if (!self.write_ok || self.stopped)
                        return false;
                    ++self.writes;
                    self.payload = payload;
                    self.timestamp = timestamp;
                    return true;
                },
                [](void* opaque) { return static_cast<Host*>(opaque)->stopped; }};
    }
};
} // namespace

TEST_CASE("Native timestamped releases sample the native clock only after draining work",
          "[ipc][video_core]") {
    Host valid;
    REQUIRE(Release::Release(0xff, 0xfffffff0, 0x12345678, 0x100002, valid.Bind()) ==
            Release::Error::None);
    REQUIRE(valid.address == 0xfffffffff0ULL);
    REQUIRE(valid.payload == 0x12345678);
    REQUIRE(valid.timestamp == 0x1122334455667788ULL);
    REQUIRE(valid.writes == 1);
    Host invalid;
    REQUIRE(Release::Release(0x100, 0, 1, 0x100002, invalid.Bind()) ==
            Release::Error::InvalidAddress);
    REQUIRE(Release::Release(0, 4, 1, 0x100002, invalid.Bind()) == Release::Error::InvalidAddress);
    REQUIRE(Release::Release(0, 0, 1, 0x1100002, invalid.Bind()) ==
            Release::Error::UnsupportedOperation);
    REQUIRE(invalid.waits == 0);
    REQUIRE(invalid.clocks == 0);
    REQUIRE(invalid.writes == 0);
}

TEST_CASE("Native timestamped release failure never publishes a payload", "[ipc][video_core]") {
    Host unreadable;
    unreadable.writable = false;
    REQUIRE(Release::Release(0, 0x1000, 1, 0x100002, unreadable.Bind()) ==
            Release::Error::MemoryFault);
    REQUIRE(unreadable.waits == 0);
    Host wait;
    wait.wait_ok = false;
    REQUIRE(Release::Release(0, 0x1000, 1, 0x100002, wait.Bind()) == Release::Error::WaitFailed);
    REQUIRE(wait.clocks == 0);
    Host write;
    write.write_ok = false;
    REQUIRE(Release::Release(0, 0x1000, 1, 0x100002, write.Bind()) == Release::Error::MemoryFault);
    REQUIRE(write.writes == 0);
    for (u32 phase : {0U, 1U, 2U}) {
        Host cancelled;
        cancelled.stopped = phase == 0;
        cancelled.cancel_in_wait = phase == 1;
        cancelled.cancel_in_write = phase == 2;
        REQUIRE(Release::Release(0, 0x1000, 1, 0x100002, cancelled.Bind()) ==
                Release::Error::Cancelled);
        REQUIRE(cancelled.writes == 0);
    }
}
