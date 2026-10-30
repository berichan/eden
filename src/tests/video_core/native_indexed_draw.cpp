// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <catch2/catch_test_macros.hpp>
#include "core/core.h"
#include "core/device_memory.h"
#include "video_core/control/native_gpu_context.h"
#include "video_core/control/native_indexed_draw.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/gpu.h"
#include "video_core/memory_manager.h"
#include "video_core/renderer_null/null_rasterizer.h"

namespace {
using Draw = Tegra::Control::NativeIndexedDraw;
using Request = Tegra::Control::NativeIndexedDrawRequest;
Draw::Registers Registers() {
    return {0x21, 0x1000, 0xff, 0xffffffff, 0, 0, 3, 0, 0, 0};
}
struct Host {
    u32 calls{};
    bool accept{true};
    bool cancelled{};
    bool cancel_during_draw{};
    std::optional<Request> request;
    Draw::Host Bind() {
        return {this,
                [](void* opaque, const Request& request) {
                    auto& self = *static_cast<Host*>(opaque);
                    ++self.calls;
                    self.request = request;
                    self.cancelled = self.cancel_during_draw;
                    return self.accept;
                },
                [](void* opaque) { return static_cast<Host*>(opaque)->cancelled; }};
    }
};
} // namespace

TEST_CASE("Native indexed spans use byte ceilings and the requested 40-bit range",
          "[ipc][video_core]") {
    auto regs = Registers();
    for (u32 format = 0; format < 3; ++format) {
        regs.format = format;
        regs.first = 7;
        const auto request = Draw::Decode(regs);
        REQUIRE(request);
        REQUIRE(request->address == 0x2100001000ULL + (7ULL << format));
        REQUIRE(request->size_bytes == (3ULL << format));
        REQUIRE(static_cast<u32>(request->format) == format);
        REQUIRE(request->count == 3);
        regs.size_upper = 0;
        regs.size_lower = 10U << format;
        REQUIRE(Draw::Decode(regs));
        --regs.size_lower;
        REQUIRE_FALSE(Draw::Decode(regs));
        regs.size_upper = 0xff;
        regs.size_lower = 0xffffffff;
    }
    regs = Registers();
    regs.address_upper = 0xff;
    regs.address_lower = 0xfffffffd;
    REQUIRE(Draw::Decode(regs));
    regs.count = 4;
    REQUIRE_FALSE(Draw::Decode(regs));
    regs.count = 1;
    regs.first = 3;
    REQUIRE_FALSE(Draw::Decode(regs));
    regs = Registers();
    regs.format = 2;
    regs.first = 0xffffffff;
    regs.count = 0xffffffff;
    REQUIRE(Draw::Decode(regs)->size_bytes == 0x3fffffffcULL);
    regs.size_upper = 0;
    REQUIRE_FALSE(Draw::Decode(regs));
    regs = Registers();
    regs.address_upper = regs.address_lower = regs.size_upper = regs.size_lower = regs.count = 0;
    REQUIRE(Draw::Decode(regs)->size_bytes == 0);
}

TEST_CASE("Native indexed spans reject unsupported control and reserved fields",
          "[ipc][video_core]") {
    for (u32 high : {0x100U, 0xffffffffU}) {
        auto regs = Registers();
        regs.address_upper = high;
        REQUIRE_FALSE(Draw::Decode(regs));
        regs = Registers();
        regs.size_upper = high;
        REQUIRE_FALSE(Draw::Decode(regs));
    }
    for (u32 format : {3U, 4U, 0xffffffffU}) {
        auto regs = Registers();
        regs.format = format;
        REQUIRE_FALSE(Draw::Decode(regs));
    }
    for (u32 value : {1U, 2U, 0xffffffffU}) {
        auto regs = Registers();
        regs.base_vertex = value;
        REQUIRE_FALSE(Draw::Decode(regs));
        regs = Registers();
        regs.base_instance = value;
        REQUIRE_FALSE(Draw::Decode(regs));
        regs = Registers();
        regs.restart = value;
        REQUIRE_FALSE(Draw::Decode(regs));
    }
}

TEST_CASE("Native indexed scopes require backend success and a matching END", "[ipc][video_core]") {
    Draw draw;
    Host host;
    REQUIRE(draw.Draw(Registers(), host.Bind()) == Draw::Error::MissingBegin);
    REQUIRE(draw.End(0) == Draw::Error::MissingBegin);
    REQUIRE(draw.Begin(5) == Draw::Error::UnsupportedBegin);
    REQUIRE_FALSE(draw.IsPending());
    REQUIRE(draw.Begin(4) == Draw::Error::None);
    REQUIRE(draw.Begin(4) == Draw::Error::NestedBegin);
    REQUIRE(draw.End(0) == Draw::Error::MissingDraw);
    REQUIRE(draw.IsPending());
    REQUIRE_FALSE(draw.HasSubmitted());
    REQUIRE(draw.Draw(Registers(), host.Bind()) == Draw::Error::None);
    REQUIRE(host.calls == 1);
    REQUIRE(host.request->address == 0x2100001000ULL);
    REQUIRE(draw.IsPending());
    REQUIRE(draw.HasSubmitted());
    REQUIRE(draw.Draw(Registers(), host.Bind()) == Draw::Error::RepeatedDraw);
    REQUIRE(host.calls == 1);
    REQUIRE(draw.End(1) == Draw::Error::UnsupportedEnd);
    REQUIRE(draw.IsPending());
    REQUIRE(draw.End(0) == Draw::Error::None);
    REQUIRE_FALSE(draw.IsPending());
    REQUIRE_FALSE(draw.HasSubmitted());
}

TEST_CASE("Failed or cancelled native draws leave their scope unfinished", "[ipc][video_core]") {
    for (u32 failure = 0; failure < 5; ++failure) {
        Draw draw;
        Host host;
        auto regs = Registers();
        REQUIRE(draw.Begin(4) == Draw::Error::None);
        host.accept = failure != 0;
        host.cancelled = failure == 1;
        host.cancel_during_draw = failure == 2;
        if (failure == 3)
            regs.size_upper = regs.size_lower = 0;
        const auto error = draw.Draw(regs, failure == 4 ? Draw::Host{} : host.Bind());
        REQUIRE(error == (failure == 1 || failure == 2 ? Draw::Error::Cancelled
                          : failure == 3               ? Draw::Error::InvalidIndexRange
                                                       : Draw::Error::DrawFailed));
        REQUIRE(draw.IsPending());
        REQUIRE_FALSE(draw.HasSubmitted());
        REQUIRE(draw.End(0) == Draw::Error::MissingDraw);
        REQUIRE(host.calls == (failure == 0 || failure == 2 ? 1U : 0U));
    }
}

TEST_CASE("Native engine never treats a null indexed backend as completed work",
          "[ipc][video_core]") {
    Core::System system;
    Core::DeviceMemory memory;
    Tegra::MaxwellDeviceMemoryManager backing{memory};
    Tegra::GPU gpu{system, false, false};
    Null::RasterizerNull rasterizer{gpu};
    Tegra::MemoryManager space{system, backing, 32, 1ULL << 32, 16, 12};
    space.BindRasterizer(&rasterizer);
    Tegra::Engines::Maxwell3D engine{space};
    engine.BindRasterizer(&rasterizer);
    engine.SetNativeMacroState(std::make_shared<Tegra::Control::AmpereGraphicsState>());
    REQUIRE(engine.CallNativeMethod(system, 0x651, 1, true, {}));
    REQUIRE(engine.CallNativeMethod(system, 0x586, 4, true, {}));
    REQUIRE(engine.HasPendingNativeDraw());
    REQUIRE(engine.CallNativeMethod(system, 0x8e, 0xff, true, {}));
    REQUIRE(engine.CallNativeMethod(system, 0x8f, 0xffffffff, true, {}));
    REQUIRE(engine.CallNativeMethod(system, 0x5f2, 0, true, {}));
    REQUIRE(engine.CallNativeMethod(system, 0x5f3, 0x8000000, true, {}));
    REQUIRE(engine.CallNativeMethod(system, 0x5f6, 0, true, {}));
    REQUIRE(engine.CallNativeMethod(system, 0x5f7, 5, true, {}));
    REQUIRE(engine.GetRegisterValue(0x5f7) == 5);
    REQUIRE(engine.draw_manager.draw_state.index_buffer.first == 0);
    REQUIRE_FALSE(engine.CallNativeMethod(system, 0x5f8, 3, true, {}));
    REQUIRE(engine.HasNativeMacroFailure());
    REQUIRE(engine.HasPendingNativeDraw());
    REQUIRE(engine.GetRegisterValue(0x5f8) == 0);
    REQUIRE(engine.draw_manager.draw_state.index_buffer.count == 0);
    REQUIRE_FALSE(engine.CallNativeMethod(system, 0x585, 0, true, {}));
    REQUIRE(engine.HasPendingNativeDraw());
}

TEST_CASE("Native suppressed draws close normally but block intervening state and reports",
          "[ipc][video_core]") {
    Core::System system;
    Core::DeviceMemory memory;
    Tegra::MaxwellDeviceMemoryManager backing{memory};
    Tegra::MemoryManager space{system, backing, 32, 1ULL << 32, 16, 12};
    for (u32 intervening : {0U, 0x8e0U, 0x6c3U}) {
        Tegra::Engines::Maxwell3D engine{space};
        engine.SetNativeMacroState(std::make_shared<Tegra::Control::AmpereGraphicsState>());
        REQUIRE(engine.CallNativeMethod(system, 0x651, 2, true, {}));
        REQUIRE(engine.CallNativeMethod(system, 0x8f, 3, true, {}));
        REQUIRE(engine.CallNativeMethod(system, 0x586, 4, true, {}));
        REQUIRE(engine.CallNativeMethod(system, 0x5f8, 3, true, {}));
        REQUIRE(engine.HasPendingNativeDraw());
        if (intervening) {
            REQUIRE_FALSE(engine.CallNativeMethod(system, intervening, 0, true, {}));
            REQUIRE(engine.HasPendingNativeDraw());
        } else {
            REQUIRE(engine.CallNativeMethod(system, 0x585, 0, true, {}));
            REQUIRE_FALSE(engine.HasPendingNativeDraw());
        }
    }
    Tegra::Engines::Maxwell3D unfinished{space};
    unfinished.SetNativeMacroState(std::make_shared<Tegra::Control::AmpereGraphicsState>());
    REQUIRE(unfinished.CallNativeMethod(system, 0x586, 4, true, {}));
    REQUIRE_FALSE(unfinished.CallNativeMethod(system, 0x6c3, 0xf006, true, {}));
    REQUIRE(unfinished.HasPendingNativeDraw());
}
