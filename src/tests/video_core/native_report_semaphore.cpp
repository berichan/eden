// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include "video_core/control/native_report_semaphore.h"

namespace {
using Semaphore = Tegra::Control::NativeReportSemaphore;
struct Host {
    GPUVAddr address{};
    u32 data{0xcccccccc};
    u32 checks{}, waits{}, writes{};
    bool writable{true}, wait_ok{true}, write_ok{true}, stopped{};
    bool cancel_in_wait{}, unmap_in_wait{};
    Semaphore::Host Bind() {
        return {this,
                [](void* opaque, GPUVAddr address, size_t size) {
                    auto& self = *static_cast<Host*>(opaque);
                    ++self.checks;
                    self.address = address;
                    REQUIRE(size == sizeof(u32));
                    return self.writable;
                },
                [](void* opaque) {
                    auto& self = *static_cast<Host*>(opaque);
                    ++self.waits;
                    if (self.cancel_in_wait)
                        self.stopped = true;
                    if (self.unmap_in_wait)
                        self.writable = false;
                    return self.wait_ok;
                },
                [](void* opaque, GPUVAddr address, u32 payload) {
                    auto& self = *static_cast<Host*>(opaque);
                    ++self.writes;
                    REQUIRE(self.waits == 1);
                    REQUIRE(address == self.address);
                    if (!self.write_ok || !self.writable)
                        return false;
                    self.data = payload;
                    return true;
                },
                [](void* opaque) { return static_cast<Host*>(opaque)->stopped; }};
    }
};
constexpr u32 ReleaseAfterWrites = 0x1000f010;
} // namespace

TEST_CASE("Native report semaphore releases retain full addresses and order writes after a drain",
          "[ipc][video_core]") {
    Host host;
    REQUIRE(Semaphore::Release(0xff, 0xfffffffc, 0x12345678, ReleaseAfterWrites, host.Bind()) ==
            Semaphore::Error::None);
    REQUIRE(host.address == 0xfffffffffcULL);
    REQUIRE(host.data == 0x12345678);
    REQUIRE(host.checks == 1);
    REQUIRE(host.waits == 1);
    REQUIRE(host.writes == 1);
    for (const auto operation : {0U, 0xf010U, 0x1000f011U, 0x1000f018U, 0x1010f010U, 0xffffffffU}) {
        Host rejected;
        REQUIRE(Semaphore::Release(0x21, 0x1000, 1, operation, rejected.Bind()) ==
                Semaphore::Error::UnsupportedOperation);
        REQUIRE(rejected.checks == 0);
        REQUIRE(rejected.waits == 0);
        REQUIRE(rejected.writes == 0);
    }
    Host upper, unaligned;
    REQUIRE(Semaphore::Release(0x100, 0, 1, ReleaseAfterWrites, upper.Bind()) ==
            Semaphore::Error::InvalidAddress);
    REQUIRE(Semaphore::Release(0, 2, 1, ReleaseAfterWrites, unaligned.Bind()) ==
            Semaphore::Error::InvalidAddress);
    REQUIRE(upper.checks == 0);
    REQUIRE(unaligned.checks == 0);
}

TEST_CASE("Native report semaphore failures never publish an uncompleted payload",
          "[ipc][video_core]") {
    Host unmapped;
    unmapped.writable = false;
    REQUIRE(Semaphore::Release(0, 0x1000, 1, ReleaseAfterWrites, unmapped.Bind()) ==
            Semaphore::Error::MemoryFault);
    REQUIRE(unmapped.waits == 0);
    REQUIRE(unmapped.writes == 0);
    Host failed_wait;
    failed_wait.wait_ok = false;
    REQUIRE(Semaphore::Release(0, 0x1000, 1, ReleaseAfterWrites, failed_wait.Bind()) ==
            Semaphore::Error::WaitFailed);
    REQUIRE(failed_wait.writes == 0);
    for (bool during_wait : {false, true}) {
        Host cancelled;
        cancelled.stopped = !during_wait;
        cancelled.cancel_in_wait = during_wait;
        REQUIRE(Semaphore::Release(0, 0x1000, 1, ReleaseAfterWrites, cancelled.Bind()) ==
                Semaphore::Error::Cancelled);
        REQUIRE(cancelled.waits == (during_wait ? 1U : 0U));
        REQUIRE(cancelled.writes == 0);
        REQUIRE(cancelled.data == 0xcccccccc);
    }
    for (bool unmap : {false, true}) {
        Host failed_write;
        failed_write.write_ok = unmap;
        failed_write.unmap_in_wait = unmap;
        REQUIRE(Semaphore::Release(0, 0x1000, 1, ReleaseAfterWrites, failed_write.Bind()) ==
                Semaphore::Error::MemoryFault);
        REQUIRE(failed_write.writes == 1);
        REQUIRE(failed_write.data == 0xcccccccc);
    }
}
