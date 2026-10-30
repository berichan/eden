// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <utility>
#include <catch2/catch_test_macros.hpp>
#include "video_core/control/native_semaphore_acquire.h"

namespace {
using Acquire = Tegra::Control::NativeSemaphoreAcquire;
struct Host {
    GPUVAddr address{};
    u32 value{5}, checks{}, waits{}, reads{};
    bool readable{true}, wait_ok{true}, read_ok{true}, stopped{}, cancel_in_wait{}, unmap_in_wait{},
        cancel_in_read{};
    Acquire::Host Bind() {
        return {this,
                [](void* opaque, GPUVAddr address, size_t size) {
                    auto& self = *static_cast<Host*>(opaque);
                    ++self.checks;
                    self.address = address;
                    REQUIRE(size == sizeof(u32));
                    return self.readable;
                },
                [](void* opaque) {
                    auto& self = *static_cast<Host*>(opaque);
                    ++self.waits;
                    if (self.cancel_in_wait)
                        self.stopped = true;
                    if (self.unmap_in_wait)
                        self.readable = false;
                    return self.wait_ok;
                },
                [](void* opaque, GPUVAddr address, u32& output) {
                    auto& self = *static_cast<Host*>(opaque);
                    ++self.reads;
                    REQUIRE(self.waits == 1);
                    REQUIRE(address == self.address);
                    if (self.cancel_in_read)
                        self.stopped = true;
                    output = self.value;
                    return self.readable && self.read_ok;
                },
                [](void* opaque) { return static_cast<Host*>(opaque)->stopped; }};
    }
};
} // namespace

TEST_CASE("Native semaphore acquire uses circular comparison without signed overflow",
          "[ipc][video_core]") {
    for (const auto [value, payload] : std::array<std::pair<u32, u32>, 5>{
             {{0, 0}, {0xffffffff, 0xffffffff}, {5, 4}, {0, 0xffffffff}, {0x7fffffff, 0}}}) {
        REQUIRE(Acquire::IsSatisfied(value, payload));
    }
    for (const auto [value, payload] :
         std::array<std::pair<u32, u32>, 5>{{{0, 1},
                                             {0xffffffff, 0},
                                             {0, 0x80000000},
                                             {0x80000000, 0},
                                             {0x7fffffff, 0xffffffff}}}) {
        REQUIRE_FALSE(Acquire::IsSatisfied(value, payload));
    }
    Host ready;
    REQUIRE(Acquire::Check(0xff, 0xfffffffc, 4, 0x1004, ready.Bind()) == Acquire::Error::None);
    REQUIRE(ready.address == 0xfffffffffcULL);
    REQUIRE(ready.reads == 1);
    Host pending;
    REQUIRE(Acquire::Check(0, 0x1000, 6, 0x1004, pending.Bind()) == Acquire::Error::NotReady);
    REQUIRE(pending.reads == 1);
    REQUIRE(pending.waits == 1);
}

TEST_CASE("Native semaphore acquire propagates invalid ranges, cancellation and backend failure",
          "[ipc][video_core]") {
    Host invalid;
    REQUIRE(Acquire::Check(0x100, 0, 0, 0x1004, invalid.Bind()) == Acquire::Error::InvalidAddress);
    REQUIRE(Acquire::Check(0, 2, 0, 0x1004, invalid.Bind()) == Acquire::Error::InvalidAddress);
    REQUIRE(Acquire::Check(0, 0, 0, 0x1005, invalid.Bind()) ==
            Acquire::Error::UnsupportedOperation);
    REQUIRE(invalid.checks == 0);
    Host unmapped;
    unmapped.readable = false;
    REQUIRE(Acquire::Check(0, 0, 0, 0x1004, unmapped.Bind()) == Acquire::Error::MemoryFault);
    REQUIRE(unmapped.waits == 0);
    Host wait;
    wait.wait_ok = false;
    REQUIRE(Acquire::Check(0, 0, 0, 0x1004, wait.Bind()) == Acquire::Error::WaitFailed);
    REQUIRE(wait.reads == 0);
    Host unmap;
    unmap.unmap_in_wait = true;
    REQUIRE(Acquire::Check(0, 0, 0, 0x1004, unmap.Bind()) == Acquire::Error::MemoryFault);
    for (u32 phase : {0U, 1U, 2U}) {
        Host cancelled;
        cancelled.stopped = phase == 0;
        cancelled.cancel_in_wait = phase == 1;
        cancelled.cancel_in_read = phase == 2;
        REQUIRE(Acquire::Check(0, 0, 0, 0x1004, cancelled.Bind()) == Acquire::Error::Cancelled);
        REQUIRE(cancelled.reads == (phase == 2 ? 1U : 0U));
    }
}
