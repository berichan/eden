// SPDX-FileCopyrightText: 2016 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <bitset>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <string>

#include "common/scope_exit.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/hle/service/psc/time/common.h"

namespace {
// Numbers are chosen randomly to make sure the correct one is given.
constexpr std::array<u64, 5> calls_order{{2, 0, 1, 4, 3}};
std::array<s64, 5> delays{};
std::bitset<5> callbacks_ran_flags;
u64 expected_callback = 0;

template <unsigned int IDX>
std::optional<std::chrono::nanoseconds> HostCallbackTemplate(s64 time,
                                                             std::chrono::nanoseconds ns_late) {
    static_assert(IDX < callbacks_ran_flags.size(), "IDX out of range");
    callbacks_ran_flags.set(IDX);
    delays[IDX] = ns_late.count();
    ++expected_callback;
    return std::nullopt;
}

struct ScopeInit final {
    ScopeInit() {
        callbacks_ran_flags.reset();
        delays.fill(0);
        expected_callback = 0;
        core_timing.SetMulticore(true);
        core_timing.Initialize([]() {});
    }

    Core::Timing::CoreTiming core_timing;
};

u64 TestTimerSpeed(Core::Timing::CoreTiming& core_timing) {
    const u64 start = core_timing.GetGlobalTimeNs().count();
    volatile u64 placebo = 0;
    for (std::size_t i = 0; i < 1000; i++) {
        placebo = placebo + core_timing.GetGlobalTimeNs().count();
    }
    const u64 end = core_timing.GetGlobalTimeNs().count();
    return end - start;
}

} // Anonymous namespace

TEST_CASE("CoreTiming[CounterProfile]", "[core][topology]") {
    const auto clock = Settings::values.cpu_clock.GetValue();
    const auto sync_speed = Settings::values.sync_core_speed.GetValue();
    const auto custom_ticks = Settings::values.use_custom_cpu_ticks.GetValue();
    SCOPE_EXIT {
        Settings::values.cpu_clock.SetValue(clock);
        Settings::values.sync_core_speed.SetValue(sync_speed);
        Settings::values.use_custom_cpu_ticks.SetValue(custom_ticks);
    };
    Settings::values.cpu_clock.SetValue(Settings::CpuClock::Normal);
    Settings::values.sync_core_speed.SetValue(false);
    Settings::values.use_custom_cpu_ticks.SetValue(false);

    using Core::Hardware::CpuProfile;
    Core::Timing::CoreTiming timing;
    timing.SetMulticore(false);
    for (auto profile : {CpuProfile::Nx, CpuProfile::Ounce, CpuProfile::Nx}) {
        timing.Initialize([] {});
        timing.SetCpuProfile(profile);
        const u64 expected = profile == CpuProfile::Ounce ? 31'250'000 : 19'200'000;
        REQUIRE(timing.GetCounterFrequency() == expected);
        REQUIRE(timing.GetClockTicks() == 0);
        timing.AddTicks(1'020'000'000);
        REQUIRE(timing.GetClockTicks() == expected);
        REQUIRE(timing.GetGlobalTimeNs() == std::chrono::seconds{1});
        // Round-trip through the actual service conversion must stay in real time.
        REQUIRE(Service::PSC::Time::ConvertToTimeSpan(timing.GetClockTicks(),
                                                      timing.GetCounterFrequency()) ==
                std::chrono::seconds{1});
    }
    timing.SetCpuProfile(CpuProfile::Ounce);
    timing.Initialize([] {});
    timing.AddTicks(32);
    REQUIRE(timing.GetClockTicks() == 0);
    timing.AddTicks(1);
    REQUIRE(timing.GetClockTicks() == 1);
    timing.Initialize([] {});
    timing.ResetTicks();
    timing.AddTicks(9'223'372'036'854'775'807ULL);
    REQUIRE(timing.GetClockTicks() == 282'578'800'148'737'003ULL);
    timing.SetMulticore(true);
    const auto before = Common::g_wall_clock.GetTimeNS().count() / 32;
    const auto counter = timing.GetClockTicks();
    const auto after = Common::g_wall_clock.GetTimeNS().count() / 32;
    REQUIRE(counter >= static_cast<u64>(before));
    REQUIRE(counter <= static_cast<u64>(after));
}

TEST_CASE("CoreTiming[CounterSpans]", "[core][topology]") {
    using Service::PSC::Time::ConvertToTimeSpan;
    for (s64 frequency : {19'200'000LL, 31'250'000LL}) {
        REQUIRE(ConvertToTimeSpan(frequency, frequency) == std::chrono::seconds{1});
        REQUIRE(ConvertToTimeSpan(-frequency, frequency) == -std::chrono::seconds{1});
        REQUIRE(ConvertToTimeSpan(0, frequency).count() == 0);
        REQUIRE(ConvertToTimeSpan(frequency + frequency / 2, frequency) ==
                std::chrono::milliseconds{1500});
        REQUIRE(ConvertToTimeSpan(-frequency - frequency / 2, frequency) ==
                -std::chrono::milliseconds{1500});
        const s64 limit = frequency * ((std::numeric_limits<s64>::max)() / 1'000'000'000);
        REQUIRE(ConvertToTimeSpan(limit, frequency).count() == 9'223'372'036'000'000'000LL);
        REQUIRE(ConvertToTimeSpan(-limit, frequency).count() == -9'223'372'036'000'000'000LL);
        REQUIRE(ConvertToTimeSpan(limit + 1, frequency) == std::chrono::nanoseconds::max());
        REQUIRE(ConvertToTimeSpan(-limit - 1, frequency) == std::chrono::nanoseconds::min());
        REQUIRE(ConvertToTimeSpan((std::numeric_limits<s64>::max)(), frequency) ==
                std::chrono::nanoseconds::max());
        REQUIRE(ConvertToTimeSpan((std::numeric_limits<s64>::min)(), frequency) ==
                std::chrono::nanoseconds::min());
    }
    REQUIRE(ConvertToTimeSpan(1, 31'250'000) == std::chrono::nanoseconds{32});
    REQUIRE(ConvertToTimeSpan(-1, 31'250'000) == std::chrono::nanoseconds{-32});
}

TEST_CASE("CoreTiming[ResetRepeatedly]", "[core]") {
    Core::Timing::CoreTiming timing;
    timing.SetMulticore(true);
    for (bool running : {false, true}) {
        for (u32 iteration = 0; iteration < 32; ++iteration) {
            timing.Initialize([] {});
            timing.SyncPause(!running);
            timing.Reset();
            REQUIRE_FALSE(timing.IsRunning());
            timing.Reset();
        }
    }
}

TEST_CASE("CoreTiming[BasicOrder]", "[core]") {
    ScopeInit guard;
    auto& core_timing = guard.core_timing;
    std::vector<std::shared_ptr<Core::Timing::EventType>> events{
        Core::Timing::CreateEvent("callbackA", HostCallbackTemplate<0>),
        Core::Timing::CreateEvent("callbackB", HostCallbackTemplate<1>),
        Core::Timing::CreateEvent("callbackC", HostCallbackTemplate<2>),
        Core::Timing::CreateEvent("callbackD", HostCallbackTemplate<3>),
        Core::Timing::CreateEvent("callbackE", HostCallbackTemplate<4>),
    };

    expected_callback = 0;

    core_timing.SyncPause(true);

    const u64 one_micro = 1000U;
    for (std::size_t i = 0; i < events.size(); i++) {
        const u64 order = calls_order[i];
        const auto future_ns = std::chrono::nanoseconds{static_cast<s64>(i * one_micro + 100)};

        core_timing.ScheduleEvent(future_ns, events[order]);
    }
    /// test pause
    REQUIRE(callbacks_ran_flags.none());

    core_timing.Pause(false); // No need to sync

    while (core_timing.HasPendingEvents())
        ;

    REQUIRE(callbacks_ran_flags.all());

    for (std::size_t i = 0; i < delays.size(); i++) {
        const double delay = static_cast<double>(delays[i]);
        const double micro = delay / 1000.0f;
        const double mili = micro / 1000.0f;
        printf("HostTimer Pausing Delay[%zu]: %.3f %.6f\n", i, micro, mili);
    }
}

TEST_CASE("CoreTiming[BasicOrderNoPausing]", "[core]") {
    ScopeInit guard;
    auto& core_timing = guard.core_timing;
    std::vector<std::shared_ptr<Core::Timing::EventType>> events{
        Core::Timing::CreateEvent("callbackA", HostCallbackTemplate<0>),
        Core::Timing::CreateEvent("callbackB", HostCallbackTemplate<1>),
        Core::Timing::CreateEvent("callbackC", HostCallbackTemplate<2>),
        Core::Timing::CreateEvent("callbackD", HostCallbackTemplate<3>),
        Core::Timing::CreateEvent("callbackE", HostCallbackTemplate<4>),
    };

    core_timing.SyncPause(true);
    core_timing.SyncPause(false);

    expected_callback = 0;

    const u64 start = core_timing.GetGlobalTimeNs().count();
    const u64 one_micro = 1000U;

    for (std::size_t i = 0; i < events.size(); i++) {
        const u64 order = calls_order[i];
        const auto future_ns = std::chrono::nanoseconds{static_cast<s64>(i * one_micro + 100)};
        core_timing.ScheduleEvent(future_ns, events[order]);
    }

    const u64 end = core_timing.GetGlobalTimeNs().count();
    const double scheduling_time = static_cast<double>(end - start);
    const double timer_time = static_cast<double>(TestTimerSpeed(core_timing));

    while (core_timing.HasPendingEvents())
        ;

    REQUIRE(callbacks_ran_flags.all());

    for (std::size_t i = 0; i < delays.size(); i++) {
        const double delay = static_cast<double>(delays[i]);
        const double micro = delay / 1000.0f;
        const double mili = micro / 1000.0f;
        printf("HostTimer No Pausing Delay[%zu]: %.3f %.6f\n", i, micro, mili);
    }

    const double micro = scheduling_time / 1000.0f;
    const double mili = micro / 1000.0f;
    printf("HostTimer No Pausing Scheduling Time: %.3f %.6f\n", micro, mili);
    printf("HostTimer No Pausing Timer Time: %.3f %.6f\n", timer_time / 1000.f,
           timer_time / 1000000.f);
}
