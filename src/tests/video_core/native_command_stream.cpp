// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <map>
#include <tuple>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include "video_core/control/native_command_stream.h"

namespace {
using Stream = Tegra::Control::NativeCommandStream;
u32 Header(u32 method, u32 count, u32 mode, u32 subchannel = 1) {
    return method | (subchannel << 13) | (count << 16) | (mode << 29);
}
u64 Entry(GPUVAddr address, size_t count, bool sync = false) {
    return address | (u64{count} << 42) | (u64{sync} << 63) | (u64{1} << 41);
}
struct Host {
    std::vector<std::tuple<u32, u32, u32, bool>> calls;
    std::map<GPUVAddr, u32> memory;
    GPUVAddr unreadable{};
    u32 bad_method{0xffffffff};
    bool cancel{};
    bool reject_barrier{};
    u32 barriers{};
    u32 read_checks{};
    Stream::Host Bind() {
        return {
            this,
            [](void* context, GPUVAddr address, size_t size) {
                auto& self = *static_cast<Host*>(context);
                ++self.read_checks;
                if (address == self.unreadable)
                    return false;
                for (size_t index = 0; index < size / 4; ++index) {
                    if (!self.memory.contains(address + index * 4))
                        return false;
                }
                return true;
            },
            [](void* context, GPUVAddr address, std::span<u32> words) {
                const auto& self = *static_cast<Host*>(context);
                for (auto& word : words) {
                    const auto it = self.memory.find(address);
                    if (it == self.memory.end())
                        return false;
                    word = it->second;
                    address += 4;
                }
                return true;
            },
            [](void* context, u32 subchannel, u32 method, u32 value, bool last) {
                auto& self = *static_cast<Host*>(context);
                if (method == self.bad_method)
                    return false;
                self.calls.emplace_back(subchannel, method, value, last);
                return true;
            },
            [](void* context) {
                auto& self = *static_cast<Host*>(context);
                ++self.barriers;
                return !self.reject_barrier;
            },
            [](void* context) { return static_cast<Host*>(context)->cancel; },
        };
    }
    void Store(GPUVAddr address, std::span<const u32> words) {
        for (const u32 word : words) {
            memory[address] = word;
            address += 4;
        }
    }
};
} // namespace

TEST_CASE("Native packets preserve increment modes and last arguments across fragments",
          "[ipc][video_core]") {
    Stream stream;
    Host host;
    const std::array first{Header(0x100, 2, 1), 4U};
    REQUIRE(stream.Process(first, 0x1000, host.Bind()).Succeeded());
    const std::array second{5U, Header(0xe00, 3, 5), 6U, 7U};
    REQUIRE(stream.Process(second, 0x2000, host.Bind()).Succeeded());
    const std::array third{8U, Header(0x222, 17, 4, 7), Header(0x300, 2, 3), 9U, 10U};
    REQUIRE(stream.Process(third, 0x3000, host.Bind()).Succeeded());
    REQUIRE(stream.Finish().Succeeded());
    const std::vector<std::tuple<u32, u32, u32, bool>> expected{
        {1, 0x100, 4, false}, {1, 0x101, 5, true},  {1, 0xe00, 6, false}, {1, 0xe01, 7, false},
        {1, 0xe01, 8, true},  {7, 0x222, 17, true}, {1, 0x300, 9, false}, {1, 0x300, 10, true},
    };
    REQUIRE(host.calls == expected);
    REQUIRE(stream.Finish().methods == expected.size());
}

TEST_CASE("Native entries validate all command ranges before the first method",
          "[ipc][video_core]") {
    Host host;
    host.Store(0x1000, std::array{Header(0x100, 1, 1), 7U});
    host.Store(0x2000, std::array{Header(0x200, 8, 4)});
    for (const u64 bad : {Entry(0x2000, 2), Entry(0x2001, 1), Entry((u64{1} << 40) - 4, 2),
                          Entry(0x2000, 1) | (u64{1} << 40), Entry(0x2000, 0)}) {
        Stream stream;
        const std::array entries{Entry(0x1000, 2), bad};
        REQUIRE_FALSE(stream.Execute(entries, host.Bind()).Succeeded());
        REQUIRE(host.calls.empty());
    }
    Stream stream;
    const std::array entries{Entry(0x1000, 2), Entry(0x2000, 1, true)};
    const auto result = stream.Execute(entries, host.Bind());
    REQUIRE(result.Succeeded());
    REQUIRE(result.methods == 2);
    REQUIRE(host.barriers == 1);
}

TEST_CASE("Native packet faults stop before later methods and remain sticky", "[ipc][video_core]") {
    Host host;
    host.bad_method = 0x101;
    Stream stream;
    const std::array words{Header(0x100, 3, 1), 1U, 2U, 3U, Header(0x300, 4, 4)};
    const auto result = stream.Process(words, 0x1000, host.Bind());
    REQUIRE(result.error == Stream::Error::MethodFailed);
    REQUIRE(result.address == 0x1008);
    REQUIRE(result.method == 0x101);
    REQUIRE(result.subchannel == 1);
    REQUIRE(result.methods == 1);
    REQUIRE(host.calls.size() == 1);
    host.bad_method = 0xffffffff;
    REQUIRE_FALSE(stream.Process(words, 0x1000, host.Bind()).Succeeded());
    REQUIRE(host.calls.size() == 1);
    for (const u32 header :
         {Header(0xfff, 2, 1), Header(0xfff, 2, 5), Header(0x1000, 1, 4), Header(0, 1, 6)}) {
        Stream rejected;
        REQUIRE_FALSE(rejected.Process(std::array{header}, 0x2000, host.Bind()).Succeeded());
        REQUIRE(rejected.Finish().methods == 0);
    }
    Stream partial;
    REQUIRE(partial.Process(std::array{Header(0x100, 2, 3), 1U}, 0x3000, host.Bind()).Succeeded());
    REQUIRE(partial.Finish().error == Stream::Error::IncompletePacket);
}

TEST_CASE("Native command decoding enforces work bounds, cancellation and real barriers",
          "[ipc][video_core]") {
    Host host;
    host.Store(0x1000, std::array{Header(0x100, 1, 4)});
    host.reject_barrier = true;
    Stream barrier;
    REQUIRE(barrier.Execute(std::array{Entry(0x1000, 1, true)}, host.Bind()).error ==
            Stream::Error::BarrierFailed);
    REQUIRE(host.calls.empty());
    host.cancel = true;
    Stream cancelled_entries;
    const u32 checks = host.read_checks;
    REQUIRE(cancelled_entries.Execute(std::array{Entry(0x1000, 1)}, host.Bind()).error ==
            Stream::Error::Cancelled);
    REQUIRE(host.read_checks == checks);
    Stream cancelled;
    REQUIRE(cancelled.Process(std::array{Header(0x100, 1, 4)}, 0x1000, host.Bind()).error ==
            Stream::Error::Cancelled);
    host.cancel = false;
    Stream many;
    const std::vector<u64> entries(Stream::MaxEntries + 1);
    REQUIRE(many.Execute(entries, host.Bind()).error == Stream::Error::InvalidEntry);
    Stream large;
    const std::array huge{
        Entry(0x1000, 0x1fffff), Entry(0x1000, 0x1fffff), Entry(0x1000, 0x1fffff),
        Entry(0x1000, 0x1fffff), Entry(0x1000, 0x1fffff), Entry(0x1000, 0x1fffff),
        Entry(0x1000, 0x1fffff), Entry(0x1000, 0x1fffff), Entry(0x1000, 0x1fffff)};
    auto permissive = host.Bind();
    permissive.can_read = [](void*, GPUVAddr, size_t) { return true; };
    REQUIRE(large.Execute(huge, permissive).error == Stream::Error::WordLimit);
}
