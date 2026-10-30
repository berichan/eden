// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <map>
#include <string>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include "video_core/control/native_qmd.h"

namespace {
using Scheduler = Tegra::Control::NativeQmdScheduler;
struct Host {
    std::map<GPUVAddr, Scheduler::Descriptor> descriptors;
    std::vector<Scheduler::Descriptor> dispatches;
    std::vector<Scheduler::Release> releases;
    std::string events;
    GPUVAddr bad_release{};
    bool reject_dispatch{};
    bool cancel_after_dispatch{};
    size_t release_limit{100000};
    bool cancelled{};
    Scheduler::Host Bind() {
        return {this,
                [](void* context, GPUVAddr address, Scheduler::Descriptor& result) {
                    auto& self = *static_cast<Host*>(context);
                    const auto it = self.descriptors.find(address);
                    if (it == self.descriptors.end())
                        return false;
                    result = it->second;
                    return true;
                },
                [](void* context, const Scheduler::Release& release) {
                    auto& self = *static_cast<Host*>(context);
                    self.events += 'V';
                    return release.address != self.bad_release && !(release.address & 3);
                },
                [](void* context, const Scheduler::Descriptor& descriptor) {
                    auto& self = *static_cast<Host*>(context);
                    self.events += 'D';
                    if (self.reject_dispatch)
                        return false;
                    self.dispatches.push_back(descriptor);
                    self.cancelled = self.cancel_after_dispatch;
                    return true;
                },
                [](void* context, const Scheduler::Release& release) {
                    auto& self = *static_cast<Host*>(context);
                    self.events += 'R';
                    if (self.releases.size() == self.release_limit)
                        return false;
                    self.releases.push_back(release);
                    return true;
                },
                [](void* context) { return static_cast<Host*>(context)->cancelled; }};
    }
};
Scheduler::Descriptor Descriptor(u32 dependencies = 1, bool nonempty = true) {
    Scheduler::Descriptor result{};
    result[4] = dependencies << 14;
    result[18] = 0x30 | (32U << 16);
    result[19] = (1U << 16) | 1;
    result[48] = 0x1234;
    if (nonempty)
        result[12] = result[13] = result[14] = 1;
    u32 index{};
    for (u32 word : {24U, 28U, 52U}) {
        result[word] = 0x3000 + index * 4;
        result[word + 1] = (1U << 23) | (1U << 30);
        result[word + 2] = 10 + index++;
    }
    return result;
}
} // namespace

TEST_CASE("Native QMD dependencies gate real dispatch and all three ordered releases",
          "[ipc][video_core]") {
    Scheduler scheduler;
    Host host;
    host.descriptors[0x1000] = Descriptor();
    auto result = scheduler.Signal(0x1000, 3, host.Bind());
    REQUIRE(result.Succeeded());
    REQUIRE(scheduler.HasPendingDispatch());
    REQUIRE(host.dispatches.empty());
    REQUIRE(host.releases.empty());
    host.descriptors[0x1000][26] = 99;
    host.descriptors[0x1000][48] = 0x5678;
    result = scheduler.Signal(0x1000, 7, host.Bind());
    REQUIRE(result.Succeeded());
    REQUIRE(result.dispatches == 1);
    REQUIRE(result.releases == 3);
    REQUIRE_FALSE(scheduler.HasPendingDispatch());
    REQUIRE(host.events == "VVVDRRR");
    REQUIRE(host.dispatches.front()[48] == 0x1234);
    REQUIRE(host.releases[0].payload == 99);
    REQUIRE(host.releases[2].address == 0x3008);
    REQUIRE(host.releases[2].SizeBytes() == 4);
}

TEST_CASE("Native QMD faults never turn skipped dispatch into a completion", "[ipc][video_core]") {
    {
        Scheduler cancelled;
        Host host;
        host.descriptors[0x1000] = Descriptor(0);
        host.cancel_after_dispatch = true;
        const auto result = cancelled.Signal(0x1000, 3, host.Bind());
        REQUIRE(result.error == Scheduler::Error::Cancelled);
        REQUIRE(result.dispatches == 1);
        REQUIRE(result.releases == 0);
        REQUIRE(host.releases.empty());
        REQUIRE(cancelled.HasFailed());
    }
    for (bool invalid_destination : {false, true}) {
        Scheduler scheduler;
        Host host;
        host.descriptors[0x1000] = Descriptor(0);
        host.reject_dispatch = !invalid_destination;
        host.bad_release = invalid_destination ? 0x3008 : 0;
        const auto result = scheduler.Signal(0x1000, 3, host.Bind());
        REQUIRE(result.error == (invalid_destination ? Scheduler::Error::InvalidRelease
                                                     : Scheduler::Error::DispatchFailed));
        REQUIRE(host.dispatches.empty());
        REQUIRE(host.releases.empty());
        const auto events = host.events;
        host.reject_dispatch = false;
        host.bad_release = 0;
        REQUIRE_FALSE(scheduler.Signal(0x1000, 3, host.Bind()).Succeeded());
        REQUIRE(host.events == events);
        REQUIRE(scheduler.HasFailed());
    }
    Scheduler scheduler;
    Host host;
    host.descriptors[0x1000] = Descriptor(0);
    host.release_limit = 1;
    const auto partial = scheduler.Signal(0x1000, 3, host.Bind());
    REQUIRE(partial.error == Scheduler::Error::ReleaseFailed);
    REQUIRE(partial.dispatches == 1);
    REQUIRE(partial.releases == 1);
    REQUIRE(host.dispatches.size() == 1);
    REQUIRE(host.releases.size() == 1);
    REQUIRE_FALSE(scheduler.Signal(0x1000, 3, host.Bind()).Succeeded());
    REQUIRE(host.dispatches.size() == 1);
}

TEST_CASE("Native QMD software counters bound deferred signals and chain dependencies",
          "[ipc][video_core]") {
    Scheduler scheduler;
    Host host;
    host.descriptors[0x1000] = Descriptor(1, false);
    REQUIRE(scheduler.Signal(0x1000, 7, host.Bind()).Succeeded());
    REQUIRE(scheduler.Signal(0x1000, 3, host.Bind()).Succeeded());
    REQUIRE(host.dispatches.empty());
    REQUIRE(host.releases.size() == 3);
    Scheduler overflow;
    constexpr u32 increment = (1023U << 14) | 12;
    for (u32 index = 0; index < 63; ++index)
        REQUIRE(overflow.Signal(0x1000, increment, {}).Succeeded());
    REQUIRE(overflow.Signal(0x1000, increment, {}).error == Scheduler::Error::CounterOverflow);
    Scheduler underflow;
    host.descriptors[0x2000] = Descriptor(0);
    REQUIRE(underflow.Signal(0x2000, 7, host.Bind()).Succeeded());
    REQUIRE(underflow.Signal(0x2000, 3, host.Bind()).error == Scheduler::Error::CounterUnderflow);
    Scheduler chain;
    host = Host{};
    host.descriptors[0x1000] = Descriptor(0);
    host.descriptors[0x1000][15] = 0x2000 >> 8;
    host.descriptors[0x1000][16] = 1 | (4U << 1);
    host.descriptors[0x2000] = Descriptor(1);
    REQUIRE(chain.Signal(0x2000, 3, host.Bind()).Succeeded());
    const auto chained = chain.Signal(0x1000, 3, host.Bind());
    REQUIRE(chained.Succeeded());
    REQUIRE(chained.dispatches == 2);
    REQUIRE(chained.releases == 6);
    REQUIRE_FALSE(chain.HasPendingDispatch());
}

TEST_CASE("Native QMD validates requests, formats and bounded traversal before visible work",
          "[ipc][video_core]") {
    for (const auto [address, argument] : std::array<std::pair<GPUVAddr, u32>, 4>{
             {{0x1001, 3}, {1ULL << 40, 3}, {0x1000, 0x100}, {0x1000, 0x1000003}}}) {
        Scheduler scheduler;
        REQUIRE(scheduler.Signal(address, argument, {}).error == Scheduler::Error::InvalidRequest);
    }
    for (u32 action : {4U, 6U, 8U, 13U}) {
        Scheduler scheduler;
        REQUIRE(scheduler.Signal(0x1000, action, {}).error == Scheduler::Error::UnsupportedAction);
    }
    for (const auto [word, bits] : std::array<std::pair<u32, u32>, 5>{
             {{18, 1}, {4, 0x100}, {4, 0x40000000}, {16, 1U << 5}, {25, 1U << 27}}}) {
        Scheduler scheduler;
        Host host;
        host.descriptors[0x1000] = Descriptor(0);
        host.descriptors[0x1000][word] |= bits;
        REQUIRE_FALSE(scheduler.Signal(0x1000, 3, host.Bind()).Succeeded());
        REQUIRE(host.dispatches.empty());
        REQUIRE(host.releases.empty());
    }
    Scheduler cancelled;
    Host host;
    host.cancelled = true;
    REQUIRE(cancelled.Signal(0x1000, 3, host.Bind()).error == Scheduler::Error::Cancelled);
    Scheduler limited;
    for (size_t index = 0; index < Scheduler::MaxDescriptors; ++index)
        REQUIRE(limited.Signal((index + 1) * 0x100, 12, {}).Succeeded());
    REQUIRE(limited.Signal((Scheduler::MaxDescriptors + 1) * 0x100, 12, {}).error ==
            Scheduler::Error::DescriptorLimit);
    Scheduler cycle;
    host = Host{};
    host.descriptors[0x1000] = Descriptor(0, false);
    host.descriptors[0x1000][15] = 0x1000 >> 8;
    host.descriptors[0x1000][16] = 1 | (3U << 1);
    const auto result = cycle.Signal(0x1000, 3, host.Bind());
    REQUIRE(result.error == Scheduler::Error::ChainLimit);
    REQUIRE(result.releases <= Scheduler::MaxChain * 3);
}

TEST_CASE("Native QMD invalidation cannot erase a dispatch waiting for dependencies",
          "[ipc][video_core]") {
    for (u32 action : {1U, 3U}) {
        Scheduler scheduler;
        Host host;
        host.descriptors[0x1000] = Descriptor(1);
        REQUIRE(scheduler.Signal(0x1000, 3, host.Bind()).Succeeded());
        REQUIRE(scheduler.HasPendingDispatch());
        REQUIRE(scheduler.Signal(0x1000, action, host.Bind()).error ==
                Scheduler::Error::UnsupportedAction);
        REQUIRE(host.dispatches.empty());
        REQUIRE(host.releases.empty());
    }
}
