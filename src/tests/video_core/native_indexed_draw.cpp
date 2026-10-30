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
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"

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

TEST_CASE("Native clipping keeps input depth range independent of pixel clamps",
          "[ipc][video_core]") {
    using Clip = Tegra::Control::NativeDepthClip;
    REQUIRE(Clip::Decode(0x181d, 0, 0, 0x3f800000)->negative_one_to_one);
    REQUIRE_FALSE(Clip::Decode(0x181d, 1, 0, 0x3f800000)->negative_one_to_one);
    REQUIRE(Clip::Decode(0x1c9d, 0, 0x80000000, 0x3f800000));
    for (u32 control : {0x181cU, 0x1815U, 0x180dU, 0x181fU, 0x1181dU, 0x8181dU, 0x101dU})
        REQUIRE_FALSE(Clip::Decode(control, 0, 0, 0x3f800000));
    REQUIRE_FALSE(Clip::Decode(0x181d, 2, 0, 0x3f800000));
    REQUIRE_FALSE(Clip::Decode(0x181d, 0, 1, 0x3f800000));
    REQUIRE_FALSE(Clip::Decode(0x181d, 0, 0, 0x3f7fffff));
}

TEST_CASE("Native blend keys validate factors, separate alpha and color masks",
          "[ipc][video_core]") {
    using State = Vulkan::FixedPipelineState;
    State::NativeBlendState input{1, 1, 0x8006, 0x4302, 0x4303, 0x8006, 0x4001, 0x4303, 0x1111};
    const auto blend = State::DecodeNativeBlend(input);
    REQUIRE(blend);
    REQUIRE(blend->enable.Value() == 1);
    REQUIRE(blend->Mask() == std::array<bool, 4>{true, true, true, true});
    auto equivalent = input;
    equivalent.color_op = equivalent.alpha_op = 1;
    equivalent.color_source = 5;
    equivalent.color_dest = equivalent.alpha_dest = 6;
    equivalent.alpha_source = 2;
    REQUIRE(State::DecodeNativeBlend(equivalent)->raw == blend->raw);
    input.separate_alpha = 0;
    input.alpha_op = input.alpha_source = input.alpha_dest = 0xffffffff;
    const auto shared = State::DecodeNativeBlend(input);
    REQUIRE(shared);
    REQUIRE(shared->equation_a.Value() == shared->equation_rgb.Value());
    REQUIRE(shared->factor_source_a.Value() == shared->factor_source_rgb.Value());
    for (u32 invalid : {0U, 12U, 13U, 16U, 0xc900U, 0xffffffffU}) {
        auto bad = input;
        bad.color_source = invalid;
        REQUIRE_FALSE(State::DecodeNativeBlend(bad));
    }
    for (u32 invalid : {0U, 6U, 0x8009U, 0xffffffffU}) {
        auto bad = input;
        bad.color_op = invalid;
        REQUIRE_FALSE(State::DecodeNativeBlend(bad));
    }
    for (u32 mask = 0; mask < 16; ++mask) {
        auto masked = input;
        masked.enabled = 0;
        masked.write_mask =
            ((mask & 1) << 0) | ((mask & 2) << 3) | ((mask & 4) << 6) | ((mask & 8) << 9);
        const auto result = State::DecodeNativeBlend(masked);
        REQUIRE(result);
        REQUIRE(result->enable.Value() == 0);
        for (u32 channel = 0; channel < 4; ++channel)
            REQUIRE(result->Mask()[channel] == bool(mask & (1U << channel)));
    }
    input.enabled = 2;
    REQUIRE_FALSE(State::DecodeNativeBlend(input));
    input.enabled = 0;
    input.write_mask = 2;
    REQUIRE_FALSE(State::DecodeNativeBlend(input));
}

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

TEST_CASE("Native pixel barriers require a real drain and reject system-memory variants",
          "[ipc][video_core]") {
    Core::System system;
    Core::DeviceMemory memory;
    Tegra::MaxwellDeviceMemoryManager backing{memory};
    Tegra::GPU gpu{system, false, false};
    Null::RasterizerNull rasterizer{gpu};
    Tegra::MemoryManager space{system, backing, 32, 1ULL << 32, 16, 12};
    space.BindRasterizer(&rasterizer);
    for (u32 value : {0U, 1U, 2U, 3U, 4U, 0xffffffffU}) {
        Tegra::Engines::Maxwell3D engine{space};
        engine.BindRasterizer(&rasterizer);
        engine.SetNativeMacroState(std::make_shared<Tegra::Control::AmpereGraphicsState>());
        REQUIRE_FALSE(engine.CallNativeMethod(system, 0x378, value, true, {}));
        REQUIRE(engine.HasNativeMacroFailure());
        REQUIRE(engine.GetRegisterValue(0x378) == 0);
    }
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

TEST_CASE("Native index bounds preserve unsigned little-endian values", "[ipc][video_core]") {
    using Bounds = Tegra::Control::NativeIndexBounds;
    const std::array<u8, 3> bytes{255, 2, 7};
    const auto eight = Bounds::Decode(Request::Format::U8, bytes);
    REQUIRE(eight);
    REQUIRE(eight->minimum == 2);
    REQUIRE(eight->maximum == 255);
    const std::array<u8, 6> shorts{0xff, 0xff, 0, 0x80, 3, 0};
    const auto sixteen = Bounds::Decode(Request::Format::U16, shorts);
    REQUIRE(sixteen);
    REQUIRE(sixteen->minimum == 3);
    REQUIRE(sixteen->maximum == 65535);
    const std::array<u8, 8> words{0xff, 0xff, 0xff, 0xff, 4, 0, 0, 0x80};
    const auto thirty_two = Bounds::Decode(Request::Format::U32, words);
    REQUIRE(thirty_two);
    REQUIRE(thirty_two->minimum == 0x80000004);
    REQUIRE(thirty_two->maximum == 0xffffffff);
    REQUIRE_FALSE(Bounds::Decode(Request::Format::U16, bytes));
    REQUIRE_FALSE(Bounds::Decode(Request::Format::U32, shorts));
    REQUIRE_FALSE(Bounds::Decode(Request::Format::U8, {}));
    REQUIRE_FALSE(Bounds::Decode(static_cast<Request::Format>(3), words));
    std::stop_source cancelled;
    cancelled.request_stop();
    REQUIRE_FALSE(Bounds::Decode(Request::Format::U8, bytes, cancelled.get_token()));
}

TEST_CASE("Native vertex fetches use byte counts and actual maximum indices", "[ipc][video_core]") {
    using Fetch = Tegra::Control::NativeVertexFetch;
    Fetch::Registers regs{0x38600000, 0x1010, 0x21, 0x1000, 0, 48, 0};
    const auto fetch = Fetch::Decode(regs, 2);
    REQUIRE(fetch);
    REQUIRE(fetch->size_bytes == 40);
    REQUIRE(fetch->address == 0x2100001000ULL);
    REQUIRE(fetch->stream == 0);
    REQUIRE(fetch->offset == 0);
    REQUIRE(fetch->stride == 16);
    REQUIRE(fetch->component_format == 3);
    REQUIRE_FALSE(Fetch::Decode(regs, 3));
    REQUIRE_FALSE(Fetch::Decode(regs, 0xffffffff));
    regs.size_lower = 40;
    REQUIRE(Fetch::Decode(regs, 2));
    regs.size_lower = 39;
    REQUIRE_FALSE(Fetch::Decode(regs, 2));
    regs.control = 0x1000;
    regs.size_lower = 8;
    REQUIRE(Fetch::Decode(regs, 0xffffffff)->size_bytes == 8);
    regs.attribute |= (6U << 7) | 31;
    regs.size_lower = 14;
    REQUIRE(Fetch::Decode(regs, 0xffffffff)->stream == 31);
    REQUIRE(Fetch::Decode(regs, 0xffffffff)->offset == 6);
    for (u32 invalid : {0x20U, 0x40U, 0x40000000U, 0x80000000U}) {
        auto bad = regs;
        bad.attribute |= invalid;
        REQUIRE_FALSE(Fetch::Decode(bad, 0));
    }
    for (u32 invalid : {0U, 0x2000U, 0x3000U}) {
        auto bad = regs;
        bad.control = invalid;
        REQUIRE_FALSE(Fetch::Decode(bad, 0));
    }
    regs.instanced = 1;
    REQUIRE_FALSE(Fetch::Decode(regs, 0));
    regs.instanced = 0;
    regs.address_upper = 0xff;
    regs.address_lower = 0xfffffff3;
    REQUIRE_FALSE(Fetch::Decode(regs, 0));
    for (const auto [format, width] : std::array<std::pair<u32, u32>, 8>{
             {{1, 16}, {2, 12}, {3, 8}, {4, 8}, {5, 6}, {15, 4}, {18, 4}, {27, 2}}}) {
        regs = {0x38000000U | (format << 21), 0x1000, 0, 0, 0, width, 0};
        REQUIRE(Fetch::Decode(regs, 0)->size_bytes == width);
    }
}
