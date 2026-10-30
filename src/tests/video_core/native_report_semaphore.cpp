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
    for (const auto operation :
         {0U, 0xf006U, 0xf010U, 0x1000f011U, 0x1000f018U, 0x1010f010U, 0xffffffffU}) {
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

namespace {
struct TimestampHost {
    GPUVAddr address{};
    u64 timestamp{};
    u32 checks{}, waits{}, clocks{}, writes{};
    bool writable{true}, wait_ok{true}, write_ok{true}, stopped{};
    bool cancel_in_wait{}, cancel_in_clock{}, cancel_in_write{}, unmap_in_wait{};
    Semaphore::TimestampHost Bind() {
        return {{this,
                 [](void* opaque, GPUVAddr address, size_t size) {
                     auto& self = *static_cast<TimestampHost*>(opaque);
                     REQUIRE(size == 16);
                     ++self.checks;
                     self.address = address;
                     return self.writable;
                 },
                 [](void* opaque) {
                     auto& self = *static_cast<TimestampHost*>(opaque);
                     ++self.waits;
                     if (self.cancel_in_wait)
                         self.stopped = true;
                     if (self.unmap_in_wait)
                         self.writable = false;
                     return self.wait_ok;
                 },
                 nullptr,
                 [](void* opaque) { return static_cast<TimestampHost*>(opaque)->stopped; }},
                [](void* opaque) -> u64 {
                    auto& self = *static_cast<TimestampHost*>(opaque);
                    REQUIRE(self.waits == 1);
                    ++self.clocks;
                    if (self.cancel_in_clock)
                        self.stopped = true;
                    return 0x0102030405060708ULL;
                },
                [](void* opaque, GPUVAddr address, u64 timestamp) {
                    auto& self = *static_cast<TimestampHost*>(opaque);
                    REQUIRE(address == self.address);
                    REQUIRE(self.clocks == 1);
                    if (self.cancel_in_write)
                        self.stopped = true;
                    if (!self.write_ok || !self.writable || self.stopped)
                        return false;
                    ++self.writes;
                    self.timestamp = timestamp;
                    return true;
                }};
    }
};
} // namespace

TEST_CASE("Native graphics timestamp reports sample nanoseconds after a real drain",
          "[ipc][video_core]") {
    TimestampHost host;
    REQUIRE(Semaphore::ReportTimestamp(0xff, 0xfffffff0, 0, 0xf006, host.Bind()) ==
            Semaphore::Error::None);
    REQUIRE(host.address == 0xfffffffff0ULL);
    REQUIRE(host.timestamp == 0x0102030405060708ULL);
    REQUIRE(host.checks == 1);
    REQUIRE(host.waits == 1);
    REQUIRE(host.clocks == 1);
    REQUIRE(host.writes == 1);
    for (const auto operation : {0U, 0xf002U, 0xf007U, 0x1000f006U, 0xa00f006U, 0xffffffffU}) {
        TimestampHost rejected;
        REQUIRE(Semaphore::ReportTimestamp(0, 0, 0, operation, rejected.Bind()) ==
                Semaphore::Error::UnsupportedOperation);
        REQUIRE(rejected.checks == 0);
        REQUIRE(rejected.waits == 0);
        REQUIRE(rejected.writes == 0);
    }
    TimestampHost nonzero, upper, unaligned;
    REQUIRE(Semaphore::ReportTimestamp(0, 0, 1, 0xf006, nonzero.Bind()) ==
            Semaphore::Error::UnsupportedOperation);
    REQUIRE(Semaphore::ReportTimestamp(0x100, 0, 0, 0xf006, upper.Bind()) ==
            Semaphore::Error::InvalidAddress);
    REQUIRE(Semaphore::ReportTimestamp(0, 8, 0, 0xf006, unaligned.Bind()) ==
            Semaphore::Error::InvalidAddress);
    REQUIRE(nonzero.checks == 0);
    REQUIRE(upper.checks == 0);
    REQUIRE(unaligned.checks == 0);
}

TEST_CASE("Native timestamp report failures never publish a partial query", "[ipc][video_core]") {
    TimestampHost unmapped;
    unmapped.writable = false;
    REQUIRE(Semaphore::ReportTimestamp(0, 0x1000, 0, 0xf006, unmapped.Bind()) ==
            Semaphore::Error::MemoryFault);
    REQUIRE(unmapped.waits == 0);
    TimestampHost wait;
    wait.wait_ok = false;
    REQUIRE(Semaphore::ReportTimestamp(0, 0x1000, 0, 0xf006, wait.Bind()) ==
            Semaphore::Error::WaitFailed);
    REQUIRE(wait.clocks == 0);
    for (bool unmap : {false, true}) {
        TimestampHost write;
        write.write_ok = unmap;
        write.unmap_in_wait = unmap;
        REQUIRE(Semaphore::ReportTimestamp(0, 0x1000, 0, 0xf006, write.Bind()) ==
                Semaphore::Error::MemoryFault);
        REQUIRE(write.writes == 0);
        REQUIRE(write.timestamp == 0);
    }
    for (u32 phase = 0; phase < 4; ++phase) {
        TimestampHost cancelled;
        cancelled.stopped = phase == 0;
        cancelled.cancel_in_wait = phase == 1;
        cancelled.cancel_in_clock = phase == 2;
        cancelled.cancel_in_write = phase == 3;
        REQUIRE(Semaphore::ReportTimestamp(0, 0x1000, 0, 0xf006, cancelled.Bind()) ==
                Semaphore::Error::Cancelled);
        REQUIRE(cancelled.writes == 0);
        REQUIRE(cancelled.timestamp == 0);
    }
    TimestampHost missing;
    auto host = missing.Bind();
    host.timestamp_ns = nullptr;
    REQUIRE(Semaphore::ReportTimestamp(0, 0x1000, 0, 0xf006, host) ==
            Semaphore::Error::MemoryFault);
    host = missing.Bind();
    host.write_checked = nullptr;
    REQUIRE(Semaphore::ReportTimestamp(0, 0x1000, 0, 0xf006, host) ==
            Semaphore::Error::MemoryFault);
    REQUIRE(missing.writes == 0);
}

TEST_CASE("Native report encodings remain specific to graphics and compute classes",
          "[ipc][video_core]") {
    Host compute;
    REQUIRE(Semaphore::Release(0xff, 0xfffffffc, 0x12345678, 0x10000000, compute.Bind(),
                               Semaphore::Engine::Compute) == Semaphore::Error::None);
    REQUIRE(compute.address == 0xfffffffffcULL);
    REQUIRE(compute.data == 0x12345678);
    Host rejected;
    REQUIRE(Semaphore::Release(0, 0, 1, 0x10000000, rejected.Bind()) ==
            Semaphore::Error::UnsupportedOperation);
    REQUIRE(Semaphore::Release(0, 0, 1, ReleaseAfterWrites, rejected.Bind(),
                               Semaphore::Engine::Compute) ==
            Semaphore::Error::UnsupportedOperation);
    REQUIRE(Semaphore::Release(0, 0, 1, 0x10000004, rejected.Bind(), Semaphore::Engine::Compute) ==
            Semaphore::Error::UnsupportedOperation);
    REQUIRE(rejected.checks == 0);
    REQUIRE(rejected.waits == 0);
    REQUIRE(rejected.writes == 0);
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
