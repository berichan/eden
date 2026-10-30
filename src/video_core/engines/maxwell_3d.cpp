// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstring>
#include <optional>
#include "common/assert.h"
#include "common/bit_util.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "video_core/control/native_constant_buffer.h"
#include "video_core/control/native_gpu_context.h"
#include "video_core/control/native_inline_upload.h"
#include "video_core/control/native_macro_processor.h"
#include "video_core/dirty_flags.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/gpu.h"
#include "video_core/memory_manager.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/textures/texture.h"

namespace Tegra::Engines {

/// First register id that is actually a Macro call.
constexpr u32 MacroRegistersStart = 0xE00;

namespace {
u32 NativeVersionRegister(u32 method) {
    switch (method) {
    case 0x5a8:
        return 0x58e; // CHECK_CLASS_VERSION / SET_CLASS_VERSION
    case 0x5aa:
        return 0x5a9; // CHECK_SPH_VERSION / SET_SPH_VERSION
    case 0x5e5:
        return 0x5e4; // CHECK_AAM_VERSION / SET_AAM_VERSION
    default:
        return 0;
    }
}
bool IsNativeVersionSet(u32 method) {
    return method == 0x58e || method == 0x5a9 || method == 0x5e4;
}
bool IsNativeConstantBufferBind(u32 method) {
    return method >= 0x904 && method <= 0x924 && (method - 0x904) % 8 == 0;
}
} // namespace

Maxwell3D::Maxwell3D(MemoryManager& memory_manager_)
    : draw_manager()
    , memory_manager{memory_manager_}
#ifdef ARCHITECTURE_x86_64
    , macro_engine(bool(Settings::values.disable_macro_jit))
#else
    , macro_engine(true)
#endif
    , upload_state{memory_manager, regs.upload}
{
    dirty.flags.flip();
    InitializeRegisterDefaults();
    execution_mask.reset();
    for (size_t i = 0; i < execution_mask.size(); i++)
        execution_mask[i] = IsMethodExecutable(u32(i));
}

Maxwell3D::~Maxwell3D() = default;

void Maxwell3D::SetNativeMacroState(std::shared_ptr<Control::AmpereGraphicsState> native_state) {
    executing_macro = 0;
    macro_params.clear();
    macro_segments.clear();
    current_macro_dirty = false;
    native_action_failed = false;
    native_instrumentation_pending = false;
    native_graphics_state = native_state;
    native_macro_processor =
        native_state ? std::make_unique<Control::NativeMacroProcessor>(std::move(native_state))
                     : nullptr;
    native_inline_upload =
        native_macro_processor ? std::make_unique<Control::NativeInlineUpload>() : nullptr;
    if (native_macro_processor) {
        execution_mask.set();
    } else {
        for (size_t method = 0; method < execution_mask.size(); ++method) {
            execution_mask[method] = IsMethodExecutable(static_cast<u32>(method));
        }
    }
}

bool Maxwell3D::HasNativeMacroFailure() const {
    return native_action_failed ||
           (native_macro_processor &&
            native_macro_processor->GetError() != Control::NativeMacroProcessor::Error::None);
}

bool Maxwell3D::HasPendingNativeMacro() const {
    return native_macro_processor && native_macro_processor->HasPendingInvocation();
}

bool Maxwell3D::HasPendingNativeInstrumentation() const {
    return native_instrumentation_pending;
}

bool Maxwell3D::HasPendingNativeInlineUpload() const {
    return native_inline_upload && native_inline_upload->IsPending();
}

const Control::NativeGraphicsConfig& Maxwell3D::GetNativeGraphicsConfig() const {
    static const Control::NativeGraphicsConfig legacy_defaults;
    return native_graphics_state ? native_graphics_state->graphics_config : legacy_defaults;
}

bool Maxwell3D::CallNativeMethod(Core::System& system, u32 method, u32 value, bool last,
                                 std::stop_token stop_token) {
    if (!native_macro_processor || stop_token.stop_requested()) {
        return false;
    }
    native_stop_token = stop_token;
    CallMethod(system, method, value, last);
    return !HasNativeMacroFailure();
}

bool Maxwell3D::IsNativeStateMethod(u32 method) {
    // C797 actions cannot fall through as state merely because the older class has
    // no executable method at the same offset. NVIDIA classes/3d/clc797.h.
    switch (method) {
    case 0x43:  // NOTIFY
    case 0x4f:  // SEND_GO_IDLE
    case 0x50:  // PM_TRIGGER
    case 0x51:  // PM_TRIGGER_WFI
    case 0x52:  // FE_ATOMIC_SEQUENCE_BEGIN
    case 0x53:  // FE_ATOMIC_SEQUENCE_END
    case 0x54:  // SET_INSTRUMENTATION_METHOD_HEADER
    case 0x55:  // SET_INSTRUMENTATION_METHOD_DATA
    case 0x5a:  // REPORT_SEMAPHORE_EXECUTE
    case 0x6c:  // LAUNCH_DMA
    case 0x6d:  // LOAD_INLINE_DATA
    case 0x80:  // RUN_DS_NOW
    case 0x87:  // INVALIDATE_SHADER_CACHES
    case 0x9a:  // DRAW_INDEX_BUFFER_BEGIN_END_A
    case 0x9b:  // DRAW_INDEX_BUFFER_BEGIN_END_B
    case 0x9c:  // DRAW_VERTEX_ARRAY_BEGIN_END_A
    case 0x9d:  // DRAW_VERTEX_ARRAY_BEGIN_END_B
    case 0x9f:  // INVALIDATE_RASTER_CACHE_NO_WFI
    case 0xb2:  // INCREMENT_SYNC_POINT
    case 0xb5:  // FLUSH_AND_INVALIDATE_ROP_MINI_CACHE
    case 0xb8:  // DECOMPRESS_SURFACE
    case 0xbc:  // PEER_SEMAPHORE_RELEASE
    case 0xbf:  // ASSIGN_ZCULL_SUBREGIONS
    case 0xc1:  // DRAW_ZERO_INDEX
    case 0xcf:  // RESERVED_TG07
    case 0xd0:  // RESERVED_TG08
    case 0xd1:  // RESERVED_TG09
    case 0xd2:  // RESERVED_TG10
    case 0xd3:  // RESERVED_TG11
    case 0xd4:  // RESERVED_TG12
    case 0xd5:  // RESERVED_TG13
    case 0xd6:  // RESERVED_TG14
    case 0xd7:  // RESERVED_TG15
    case 0x142: // LOAD_ROOT_TABLE
    case 0x157: // MME_DMA_READ
    case 0x158: // MME_DMA_READ_FIFOED
    case 0x159: // MME_DMA_WRITE
    case 0x15a: // MME_DMA_REDUCTION
    case 0x15b: // MME_DMA_SYSMEMBAR
    case 0x15c: // MME_DMA_SYNC
    case 0x1ff: // THROTTLE_SM
    case 0x35e: // DRAW_VERTEX_ARRAY
    case 0x366: // FORCE_TRANSITION_TO_BETA
    case 0x369: // INVALIDATE_SHADER_CACHES_NO_WFI
    case 0x378: // PIXEL_SHADER_BARRIER
    case 0x37b: // MME_DMA_WRITE_METHOD_BARRIER
    case 0x37c: // LOAD_ITERATED_BLEND_INSTRUCTION_POINTER
    case 0x37d: // LOAD_ITERATED_BLEND_INSTRUCTION
    case 0x3c6: // PM_LOCAL_TRIGGER
    case 0x43d: // UNBIND_ALL
    case 0x445: // PM_TRIGGER_END
    case 0x451: // FLUSH_PENDING_WRITES
    case 0x47d: // FORCE_HEAVYWEIGHT_METHOD_SYNC
    case 0x47f: // DECOMPRESS_ZETA_SURFACE
    case 0x485: // DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_FIRST
    case 0x486: // DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_SUBSEQUENT
    case 0x48f: // DRAW_AUTO
    case 0x4a2: // INVALIDATE_TEXTURE_DATA_CACHE_NO_WFI
    case 0x4b2: // CLEAR_ZCULL_REGION
    case 0x4c1: // DRAW_INLINE_INDEX4X8
    case 0x4cc: // INVALIDATE_SAMPLER_CACHE
    case 0x4cd: // INVALIDATE_TEXTURE_HEADER_CACHE
    case 0x4ce: // INVALIDATE_TEXTURE_DATA_CACHE
    case 0x509: // INVALIDATE_SAMPLER_CACHE_NO_WFI
    case 0x50a: // INVALIDATE_TEXTURE_HEADER_CACHE_NO_WFI
    case 0x519: // STORE_ZCULL
    case 0x540: // LOAD_ZCULL
    case 0x54c: // CLEAR_REPORT_VALUE
    case 0x57a: // DRAW_INLINE_INDEX
    case 0x57c: // DRAW_INLINE_INDEX2X16
    case 0x585: // END
    case 0x586: // BEGIN
    case 0x588: // ADD_TO_PRIMITIVE_ID
    case 0x589: // LOAD_PRIMITIVE_ID
    case 0x5a8: // CHECK_CLASS_VERSION
    case 0x5aa: // CHECK_SPH_VERSION
    case 0x5e5: // CHECK_AAM_VERSION
    case 0x5f8: // DRAW_INDEX_BUFFER
    case 0x5f9: // DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_FIRST
    case 0x5fa: // DRAW_INDEX_BUFFER16_BEGIN_END_INSTANCE_FIRST
    case 0x5fb: // DRAW_INDEX_BUFFER8_BEGIN_END_INSTANCE_FIRST
    case 0x5fc: // DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_SUBSEQUENT
    case 0x5fd: // DRAW_INDEX_BUFFER16_BEGIN_END_INSTANCE_SUBSEQUENT
    case 0x5fe: // DRAW_INDEX_BUFFER8_BEGIN_END_INSTANCE_SUBSEQUENT
    case 0x656: // INVALIDATE_ZCULL
    case 0x65e: // ZCULL_SYNC
    case 0x674: // CLEAR_SURFACE
    case 0x675: // CLEAR_CLIP_ID_SURFACE
    case 0x6c3: // SET_REPORT_SEMAPHORE_D
    case 0xccd: // ENABLE_SHADER_PERFORMANCE_SNAPSHOT_COUNTER
    case 0xcce: // DISABLE_SHADER_PERFORMANCE_SNAPSHOT_COUNTER
    case 0xcf8: // START_SHADER_PERFORMANCE_COUNTER
    case 0xcf9: // STOP_SHADER_PERFORMANCE_COUNTER
        return false;
    default:
        break;
    }
    return !(method >= 0x154 && method <= 0x15d) && !(method >= 0x8e4 && method <= 0x8f3) &&
           !(method >= 0x499 && method <= 0x49d) &&
           (!IsMethodExecutable(method) || method == MAXWELL3D_REG_INDEX(shadow_ram_control));
}

bool Maxwell3D::NativeWaitForIdle() {
    return !native_stop_token.stop_requested() && rasterizer &&
           rasterizer->SynchronizeNativeWork() && !native_stop_token.stop_requested();
}

void Maxwell3D::BindRasterizer(VideoCore::RasterizerInterface* rasterizer_) {
    rasterizer = rasterizer_;
    upload_state.BindRasterizer(rasterizer_);
}

void Maxwell3D::InitializeRegisterDefaults() {
    // Initializes registers to their default values - what games expect them to be at boot. This is
    // for certain registers that may not be explicitly set by games.

    // Reset all registers to zero
    std::memset(&regs, 0, sizeof(regs));

    // Depth range near/far is not always set, but is expected to be the default 0.0f, 1.0f. This is
    // needed for ARMS.
    for (auto& viewport : regs.viewports) {
        viewport.depth_range_near = 0.0f;
        viewport.depth_range_far = 1.0f;
    }
    for (auto& viewport : regs.viewport_transform) {
        viewport.swizzle.x.Assign(Regs::ViewportSwizzle::PositiveX);
        viewport.swizzle.y.Assign(Regs::ViewportSwizzle::PositiveY);
        viewport.swizzle.z.Assign(Regs::ViewportSwizzle::PositiveZ);
        viewport.swizzle.w.Assign(Regs::ViewportSwizzle::PositiveW);
    }

    // Doom and Bomberman seems to use the uninitialized registers and just enable blend
    // so initialize blend registers with sane values
    regs.blend.color_op = Regs::Blend::Equation::Add_D3D;
    regs.blend.color_source = Regs::Blend::Factor::One_D3D;
    regs.blend.color_dest = Regs::Blend::Factor::Zero_D3D;
    regs.blend.alpha_op = Regs::Blend::Equation::Add_D3D;
    regs.blend.alpha_source = Regs::Blend::Factor::One_D3D;
    regs.blend.alpha_dest = Regs::Blend::Factor::Zero_D3D;
    for (auto& blend : regs.blend_per_target) {
        blend.color_op = Regs::Blend::Equation::Add_D3D;
        blend.color_source = Regs::Blend::Factor::One_D3D;
        blend.color_dest = Regs::Blend::Factor::Zero_D3D;
        blend.alpha_op = Regs::Blend::Equation::Add_D3D;
        blend.alpha_source = Regs::Blend::Factor::One_D3D;
        blend.alpha_dest = Regs::Blend::Factor::Zero_D3D;
    }
    regs.stencil_front_op.fail = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_front_op.zfail = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_front_op.zpass = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_front_op.func = Regs::ComparisonOp::Always_GL;
    regs.stencil_front_func_mask = 0xFFFFFFFF;
    regs.stencil_front_mask = 0xFFFFFFFF;
    regs.stencil_two_side_enable = 1;
    regs.stencil_back_op.fail = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_back_op.zfail = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_back_op.zpass = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_back_op.func = Regs::ComparisonOp::Always_GL;
    regs.stencil_back_func_mask = 0xFFFFFFFF;
    regs.stencil_back_mask = 0xFFFFFFFF;

    regs.depth_test_func = Regs::ComparisonOp::Always_GL;
    regs.gl_front_face = Regs::FrontFace::CounterClockWise;
    regs.gl_cull_face = Regs::CullFace::Back;

    // TODO(Rodrigo): Most games do not set a point size. I think this is a case of a
    // register carrying a default value. Assume it's OpenGL's default (1).
    regs.point_size = 1.0f;

    // TODO(bunnei): Some games do not initialize the color masks (e.g. Sonic Mania). Assuming a
    // default of enabled fixes rendering here.
    for (auto& color_mask : regs.color_mask) {
        color_mask.R.Assign(1);
        color_mask.G.Assign(1);
        color_mask.B.Assign(1);
        color_mask.A.Assign(1);
    }

    for (auto& format : regs.vertex_attrib_format) {
        format.constant.Assign(1);
    }

    // NVN games expect these values to be enabled at boot
    regs.rasterize_enable = 1;
    regs.color_target_mrt_enable = 1;
    regs.framebuffer_srgb = 1;
    regs.line_width_aliased = 1.0f;
    regs.line_width_smooth = 1.0f;
    regs.gl_front_face = Maxwell3D::Regs::FrontFace::ClockWise;
    regs.polygon_mode_back = Maxwell3D::Regs::PolygonMode::Fill;
    regs.polygon_mode_front = Maxwell3D::Regs::PolygonMode::Fill;

    shadow_state = regs;
}

bool Maxwell3D::IsMethodExecutable(u32 method) {
    if (method >= MacroRegistersStart) {
        return true;
    }
    switch (method) {
    case MAXWELL3D_REG_INDEX(draw.end):
    case MAXWELL3D_REG_INDEX(draw.begin):
    case MAXWELL3D_REG_INDEX(vertex_buffer.first):
    case MAXWELL3D_REG_INDEX(vertex_buffer.count):
    case MAXWELL3D_REG_INDEX(index_buffer.first):
    case MAXWELL3D_REG_INDEX(index_buffer.count):
    case MAXWELL3D_REG_INDEX(draw_inline_index):
    case MAXWELL3D_REG_INDEX(index_buffer32_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer16_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer8_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer32_first):
    case MAXWELL3D_REG_INDEX(index_buffer16_first):
    case MAXWELL3D_REG_INDEX(index_buffer8_first):
    case MAXWELL3D_REG_INDEX(inline_index_2x16.even):
    case MAXWELL3D_REG_INDEX(inline_index_4x8.index0):
    case MAXWELL3D_REG_INDEX(vertex_array_instance_first):
    case MAXWELL3D_REG_INDEX(vertex_array_instance_subsequent):
    case MAXWELL3D_REG_INDEX(draw_texture.src_y0):
    case MAXWELL3D_REG_INDEX(wait_for_idle):
    case MAXWELL3D_REG_INDEX(shadow_ram_control):
    case MAXWELL3D_REG_INDEX(load_mme.instruction_ptr):
    case MAXWELL3D_REG_INDEX(load_mme.instruction):
    case MAXWELL3D_REG_INDEX(load_mme.start_address):
    case MAXWELL3D_REG_INDEX(falcon[4]):
    case MAXWELL3D_REG_INDEX(const_buffer.buffer):
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 1:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 2:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 3:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 4:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 5:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 6:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 7:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 8:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 9:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 10:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 11:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 12:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 13:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 14:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 15:
    case MAXWELL3D_REG_INDEX(bind_groups[0].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[1].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[2].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[3].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[4].raw_config):
    case MAXWELL3D_REG_INDEX(topology_override):
    case MAXWELL3D_REG_INDEX(clear_surface):
    case MAXWELL3D_REG_INDEX(report_semaphore.query):
    case MAXWELL3D_REG_INDEX(render_enable.mode):
    case MAXWELL3D_REG_INDEX(clear_report_value):
    case MAXWELL3D_REG_INDEX(sync_info):
    case MAXWELL3D_REG_INDEX(launch_dma):
    case MAXWELL3D_REG_INDEX(inline_data):
    case MAXWELL3D_REG_INDEX(fragment_barrier):
    case MAXWELL3D_REG_INDEX(invalidate_texture_data_cache):
    case MAXWELL3D_REG_INDEX(tiled_cache_barrier):
        return true;
    default:
        return false;
    }
}

void Maxwell3D::ProcessMacro(Core::System& system, u32 method, const u32* base_start, u32 amount, bool is_last_call) {
    if (executing_macro == 0) {
        // A macro call must begin by writing the macro method's register, not its argument.
        ASSERT((method % 2) == 0 && "Can't start macro execution by writing to the ARGS register");
        executing_macro = method;
    }

    macro_params.insert(macro_params.end(), base_start, base_start + amount);
    macro_segments.emplace_back(current_dma_segment, amount);
    current_macro_dirty |= current_dirty;
    current_dirty = false;

    // Call the macro when there are no more parameters in the command buffer
    if (is_last_call) {
        ConsumeSink(system);
        CallMacroMethod(system, executing_macro, macro_params);
        macro_params.clear();
        macro_segments.clear();
        current_macro_dirty = false;
    }
}

void Maxwell3D::RefreshParametersImpl() {
    if (!Settings::IsGPULevelHigh()) {
        return;
    }
    size_t current_index = 0;
    for (auto& segment : macro_segments) {
        if (segment.first == 0) {
            current_index += segment.second;
            continue;
        }
        memory_manager.ReadBlock(segment.first, &macro_params[current_index],
                                 sizeof(u32) * segment.second);
        current_index += segment.second;
    }
}

u32 Maxwell3D::GetMaxCurrentVertices() {
    u32 num_vertices = 0;
    for (size_t index = 0; index < Regs::NumVertexArrays; ++index) {
        const auto& array = regs.vertex_streams[index];
        if (array.enable == 0) {
            continue;
        }
        const auto& attribute = regs.vertex_attrib_format[index];
        if (attribute.constant) {
            num_vertices = (std::max)(num_vertices, 1U);
            continue;
        }
        const auto& limit = regs.vertex_stream_limits[index];
        const GPUVAddr gpu_addr_begin = array.Address();
        const GPUVAddr gpu_addr_end = limit.Address() + 1;
        const u32 address_size = static_cast<u32>(gpu_addr_end - gpu_addr_begin);
        num_vertices = (std::max)(
            num_vertices, address_size / (std::max)(attribute.SizeInBytes(), array.stride.Value()));
        break;
    }
    return num_vertices;
}

size_t Maxwell3D::EstimateIndexBufferSize() {
    GPUVAddr start_address = regs.index_buffer.StartAddress();
    GPUVAddr end_address = regs.index_buffer.EndAddress();
    auto const byte_size = regs.index_buffer.FormatSizeInBytes();
    auto const max_size = 1ull << (byte_size * CHAR_BIT);
    auto const upper_cap = GetMaxCurrentVertices() * 4 * byte_size;
    auto const lower_cap = std::min<size_t>(size_t(end_address - start_address), upper_cap);
    return std::min<size_t>(memory_manager.GetMemoryLayoutSize(start_address, byte_size * max_size) / byte_size, lower_cap);
}

u32 Maxwell3D::ProcessShadowRam(u32 method, u32 argument) {
    // Keep track of the register value in shadow_state when requested.
    auto const c = shadow_state.shadow_ram_control;
    if (c == Regs::ShadowRamControl::Track || c == Regs::ShadowRamControl::TrackWithFilter)
        return shadow_state.reg_array[method] = argument;
    else if (c == Regs::ShadowRamControl::Replay)
        return shadow_state.reg_array[method];
    return argument;
}

void Maxwell3D::ConsumeSinkImpl(Core::System& system) {
    const auto control = shadow_state.shadow_ram_control;
    if (control == Regs::ShadowRamControl::Track || control == Regs::ShadowRamControl::TrackWithFilter) {
        for (auto [method, value] : method_sink) {
            shadow_state.reg_array[method] = value;
            ProcessDirtyRegisters(method, value);
        }
    } else if (control == Regs::ShadowRamControl::Replay) {
        for (auto [method, value] : method_sink)
            ProcessDirtyRegisters(method, shadow_state.reg_array[method]);
    } else {
        for (auto [method, value] : method_sink)
            ProcessDirtyRegisters(method, value);
    }
    method_sink.clear();
}

void Maxwell3D::ProcessDirtyRegisters(u32 method, u32 argument) {
    regs.reg_array[method] = argument;
    for (auto const& table : dirty.tables)
        dirty.flags[table[method]] = true;
}

void Maxwell3D::ProcessMethodCall(u32 method, u32 argument, u32 nonshadow_argument, bool is_last_call) {
    switch (method) {
    case MAXWELL3D_REG_INDEX(wait_for_idle):
        return rasterizer->WaitForIdle();
    case MAXWELL3D_REG_INDEX(shadow_ram_control):
        shadow_state.shadow_ram_control = static_cast<Regs::ShadowRamControl>(nonshadow_argument);
        return;
    case MAXWELL3D_REG_INDEX(load_mme.instruction_ptr):
        return macro_engine.ClearCode(regs.load_mme.instruction_ptr);
    case MAXWELL3D_REG_INDEX(load_mme.instruction):
        return macro_engine.AddCode(regs.load_mme.instruction_ptr, argument);
    case MAXWELL3D_REG_INDEX(load_mme.start_address):
        return ProcessMacroBind(argument);
    case MAXWELL3D_REG_INDEX(falcon[4]):
        return ProcessFirmwareCall4();
    case MAXWELL3D_REG_INDEX(const_buffer.buffer):
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 1:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 2:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 3:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 4:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 5:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 6:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 7:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 8:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 9:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 10:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 11:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 12:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 13:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 14:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 15:
        return ProcessCBData(argument);
    case MAXWELL3D_REG_INDEX(bind_groups[0].raw_config):
        return ProcessCBBind(0);
    case MAXWELL3D_REG_INDEX(bind_groups[1].raw_config):
        return ProcessCBBind(1);
    case MAXWELL3D_REG_INDEX(bind_groups[2].raw_config):
        return ProcessCBBind(2);
    case MAXWELL3D_REG_INDEX(bind_groups[3].raw_config):
        return ProcessCBBind(3);
    case MAXWELL3D_REG_INDEX(bind_groups[4].raw_config):
        return ProcessCBBind(4);
    case MAXWELL3D_REG_INDEX(report_semaphore.query):
        return ProcessQueryGet();
    case MAXWELL3D_REG_INDEX(render_enable.mode):
        return ProcessQueryCondition();
    case MAXWELL3D_REG_INDEX(clear_report_value):
        return ProcessCounterReset();
    case MAXWELL3D_REG_INDEX(sync_info):
        return ProcessSyncPoint();
    case MAXWELL3D_REG_INDEX(launch_dma):
        return upload_state.ProcessExec(regs.launch_dma.memory_layout.Value() == Regs::LaunchDMA::Layout::Pitch);
    case MAXWELL3D_REG_INDEX(inline_data):
        upload_state.ProcessData(argument, is_last_call);
        return;
    case MAXWELL3D_REG_INDEX(fragment_barrier):
        return rasterizer->FragmentBarrier();
    case MAXWELL3D_REG_INDEX(invalidate_texture_data_cache):
        rasterizer->InvalidateGPUCache();
        return rasterizer->WaitForIdle();
    case MAXWELL3D_REG_INDEX(tiled_cache_barrier):
        return rasterizer->TiledCacheBarrier();
    default:
        draw_manager.ProcessMethodCall(*this, method, argument);
        break;
    }
}

void Maxwell3D::CallMacroMethod(Core::System& system, u32 method, const std::vector<u32>& parameters) {
    // Reset the current macro.
    executing_macro = 0;

    // Lookup the macro offset
    const u32 entry =
        ((method - MacroRegistersStart) >> 1) % static_cast<u32>(macro_positions.size());

    // Execute the current macro.
    macro_engine.Execute(system, *this, macro_positions[entry], parameters);
    draw_manager.DrawDeferred(*this);
}

void Maxwell3D::CallMethod(Core::System& system, u32 method, u32 method_argument, bool is_last_call) {
    if (HasNativeMacroFailure()) {
        return;
    }
    if (native_macro_processor) {
        if (method >= MacroRegistersStart || (method >= 0x45 && method <= 0x48)) {
            ConsumeSink(system);
        }
        struct HostContext {
            Maxwell3D& engine;
            Core::System& system;
        } context{*this, system};
        const Control::NativeMacroProcessor::Host host{
            &context,
            [](void* context, u32 method, u32& value) {
                value = static_cast<HostContext*>(context)->engine.GetRegisterValue(method);
                return method != 0x8e3 || value <= 0xffff;
            },
            [](void* context, u32 method, u32 value) {
                auto& self = *static_cast<HostContext*>(context);
                if (method == 0x44) { // C797 WAIT_FOR_IDLE
                    return self.engine.NativeWaitForIdle();
                }
                if (NativeVersionRegister(method) || IsNativeVersionSet(method) ||
                    IsNativeConstantBufferBind(method) || method == 0x8c4 || method == 0x54 ||
                    method == 0x55 || method == 0x6c || method == 0x6d ||
                    (method >= 0x8e4 && method <= 0x8f3)) {
                    self.engine.CallMethod(self.system, method, value, true);
                    return !self.engine.HasNativeMacroFailure();
                }
                // Native draw/action adapters must opt in before MME can execute those methods.
                if (!self.engine.IsNativeStateMethod(method)) {
                    LOG_ERROR(HW_GPU,
                              "Unsupported native graphics macro output method={:#x}, value={:#x}",
                              method, value);
                    return false;
                }
                self.engine.CallMethod(self.system, method, value, true);
                return true;
            },
            [](void* context) {
                const auto& self = static_cast<HostContext*>(context)->engine;
                return self.rasterizer && self.rasterizer->SynchronizeNativeWork();
            },
            [](void* context) {
                return static_cast<HostContext*>(context)
                    ->engine.native_stop_token.stop_requested();
            },
        };
        const auto status =
            native_macro_processor->CallMethod(method, method_argument, is_last_call, host);
        if (status != Control::NativeMacroProcessor::Status::Unhandled) {
            if (status == Control::NativeMacroProcessor::Status::Failed) {
                const auto& execution = native_macro_processor->GetExecutionResult();
                LOG_ERROR(HW_GPU,
                          "Native graphics macro failed: method={:#x}, error={}, "
                          "execution_error={}, pc={:#x}",
                          method, static_cast<u32>(native_macro_processor->GetError()),
                          static_cast<u32>(execution.error), execution.program_counter);
            }
            if (status == Control::NativeMacroProcessor::Status::Accepted && method >= 0x45 &&
                method <= 0x48) {
                ProcessDirtyRegisters(method, method_argument);
            }
            return;
        }
        if (method == 0x44) {
            native_action_failed = !NativeWaitForIdle();
            return;
        }
        if (IsNativeConstantBufferBind(method)) {
            const u32 slot = (method_argument >> 4) & 0x1f;
            if ((method_argument & ~0x1f1U) || slot >= Regs::MaxConstBuffers) {
                native_action_failed = true;
                return;
            }
            ConstBufferInfo binding{};
            binding.enabled = (method_argument & 1) != 0;
            if (binding.enabled) {
                const u32 upper = GetRegisterValue(0x8e1);
                binding.address = (GPUVAddr{upper} << 32) | GetRegisterValue(0x8e2);
                binding.size = GetRegisterValue(0x8e0);
                if (upper > 0xff || binding.size == 0 || binding.size > 0x1ffff ||
                    binding.size > (u64{1} << 40) - binding.address ||
                    !memory_manager.IsReadableRange(binding.address, binding.size)) {
                    native_action_failed = true;
                    return;
                }
            }
            // Preserve canonical GPUVA state. The legacy cache binder assumes one
            // contiguous physical mapping; native draws remain guarded until resolved.
            state.shader_stages[(method - 0x904) / 8].const_buffers[slot] = binding;
            ProcessDirtyRegisters(method, method_argument);
            return;
        }
        if (method == 0x6c || method == 0x6d) {
            if (native_instrumentation_pending) {
                native_action_failed = true;
                return;
            }
            using Transfer = Control::NativeInlineUpload;
            const Transfer::Host upload_host{
                this,
                [](void* context, GPUVAddr address, size_t size) {
                    return static_cast<Maxwell3D*>(context)->memory_manager.IsWritableRange(address,
                                                                                            size);
                },
                [](void* context) { return static_cast<Maxwell3D*>(context)->NativeWaitForIdle(); },
                [](void* context, GPUVAddr address, std::span<const u8> bytes) {
                    return static_cast<Maxwell3D*>(context)->memory_manager.WriteBlockChecked(
                        address, bytes);
                },
                [](void* context) {
                    return static_cast<Maxwell3D*>(context)->native_stop_token.stop_requested();
                },
            };
            const auto error =
                method == 0x6c
                    ? native_inline_upload->Launch({GetRegisterValue(0x60), GetRegisterValue(0x61),
                                                    GetRegisterValue(0x62), GetRegisterValue(0x63),
                                                    GetRegisterValue(0x64)},
                                                   method_argument, upload_host)
                    : native_inline_upload->PushWord(method_argument, upload_host);
            if (error != Transfer::Error::None) {
                LOG_ERROR(HW_GPU,
                          "Native inline DMA failed: method={:#x}, argument={:#x}, error={}, "
                          "bytes={:#x}, lines={:#x}",
                          method, method_argument, static_cast<u32>(error), GetRegisterValue(0x60),
                          GetRegisterValue(0x61));
                native_action_failed = true;
                return;
            }
            ProcessDirtyRegisters(method, method_argument);
            return;
        }
        if (method == 0x54 || method == 0x55) {
            // Retain only the initialization metadata pair used by deko3d and the
            // native stream. Profiling and hardware method forwarding are unsupported.
            const bool header = method == 0x54;
            if (HasPendingNativeInlineUpload() || native_instrumentation_pending == header ||
                method_argument != (header ? 0x49000000U : 0x49000001U)) {
                native_action_failed = true;
                return;
            }
            native_instrumentation_pending = header;
            ProcessDirtyRegisters(method, method_argument);
            return;
        }
        if (method >= 0x8e4 && method <= 0x8f3) {
            using Buffer = Control::NativeConstantBuffer;
            Buffer::Selection selection{GetRegisterValue(0x8e0), GetRegisterValue(0x8e1),
                                        GetRegisterValue(0x8e2), GetRegisterValue(0x8e3)};
            const Buffer::Host upload_host{
                this,
                [](void* context, GPUVAddr address, size_t size) {
                    return static_cast<Maxwell3D*>(context)->memory_manager.IsWritableRange(address,
                                                                                            size);
                },
                [](void* context) { return static_cast<Maxwell3D*>(context)->NativeWaitForIdle(); },
                [](void* context, GPUVAddr address, std::span<const u32> words) {
                    return static_cast<Maxwell3D*>(context)->memory_manager.WriteBlockChecked(
                        address, {reinterpret_cast<const u8*>(words.data()), words.size_bytes()});
                },
                [](void* context) {
                    return static_cast<Maxwell3D*>(context)->native_stop_token.stop_requested();
                },
            };
            const auto error =
                Buffer::Upload(selection, std::span{&method_argument, 1}, upload_host);
            if (error != Buffer::Error::None) {
                LOG_ERROR(HW_GPU, "Native constant-buffer upload failed: method={:#x}, error={}",
                          method, static_cast<u32>(error));
                native_action_failed = true;
                return;
            }
            ProcessDirtyRegisters(method, method_argument);
            ProcessDirtyRegisters(0x8e3, selection.offset);
            return;
        }
        if (method == 0x8c4) {
            const Control::NativeGraphicsConfig::Host firmware_host{
                this,
                [](void* context) { return static_cast<Maxwell3D*>(context)->NativeWaitForIdle(); },
                [](void* context) {
                    return static_cast<Maxwell3D*>(context)->native_stop_token.stop_requested();
                },
            };
            const auto error = native_graphics_state->graphics_config.ApplyMaskedRegister(
                method_argument, GetRegisterValue(0xd01), GetRegisterValue(0xd02), firmware_host);
            if (error != Control::NativeGraphicsConfig::Error::None) {
                LOG_ERROR(HW_GPU,
                          "Native Falcon request rejected: target={:#x}, value={:#x}, mask={:#x}, "
                          "error={}",
                          method_argument, GetRegisterValue(0xd01), GetRegisterValue(0xd02),
                          static_cast<u32>(error));
                native_action_failed = true;
                return;
            }
            // The modeled context setting is applied before acknowledging the firmware method.
            // Draw actions remain unsupported until the rasterizer honors its NaN policy.
            ProcessDirtyRegisters(method, method_argument);
            ProcessDirtyRegisters(0xd00, 1);
            return;
        }
        if (const u32 version_register = NativeVersionRegister(method)) {
            const u32 configured = GetRegisterValue(version_register);
            // Support exact agreement with a declared, nonempty version interval.
            native_action_failed = configured == 0 || configured != method_argument ||
                                   (configured >> 16) > (configured & 0xffff);
            return;
        }
        if (IsNativeVersionSet(method)) {
            if (!method_argument || (method_argument >> 16) > (method_argument & 0xffff)) {
                native_action_failed = true;
            } else {
                ProcessDirtyRegisters(method, method_argument);
            }
            return;
        }
        if (!IsNativeStateMethod(method)) {
            native_action_failed = true;
            return;
        }
    }
    // It is an error to write to a register other than the current macro's ARG register before
    // it has finished execution.
    if (executing_macro != 0) {
        ASSERT(method == executing_macro + 1);
    }

    // Methods after 0xE00 are special, they're actually triggers for some microcode that was
    // uploaded to the GPU during initialization.
    if (method >= MacroRegistersStart) {
        ProcessMacro(system, method, &method_argument, 1, is_last_call);
        return;
    }

    ASSERT(method < Regs::NUM_REGS && "Invalid Maxwell3D register, increase the size of the Regs structure");
    const u32 argument = ProcessShadowRam(method, method_argument);
    ProcessDirtyRegisters(method, argument);
    ProcessMethodCall(method, argument, method_argument, is_last_call);
}

void Maxwell3D::CallMultiMethod(Core::System& system, u32 method, const u32* base_start, u32 amount, u32 methods_pending) {
    if (native_macro_processor) {
        for (u32 index = 0; index < amount; ++index) {
            CallMethod(system, method, base_start[index], methods_pending - index <= 1);
        }
        return;
    }
    // Methods after 0xE00 are special, they're actually triggers for some microcode that was
    // uploaded to the GPU during initialization.
    if (method >= MacroRegistersStart) {
        ProcessMacro(system, method, base_start, amount, amount == methods_pending);
        return;
    }
    switch (method) {
    case MAXWELL3D_REG_INDEX(const_buffer.buffer):
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 1:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 2:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 3:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 4:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 5:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 6:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 7:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 8:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 9:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 10:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 11:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 12:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 13:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 14:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 15:
        ProcessCBMultiData(base_start, amount);
        break;
    case MAXWELL3D_REG_INDEX(inline_data): {
        ASSERT(methods_pending == amount);
        upload_state.ProcessData(base_start, amount);
        return;
    }
    case MAXWELL3D_REG_INDEX(draw_inline_index):
    case MAXWELL3D_REG_INDEX(inline_index_2x16.even):
    case MAXWELL3D_REG_INDEX(inline_index_4x8.index0):
        if (shadow_state.shadow_ram_control != Regs::ShadowRamControl::Replay) {
            ProcessInlineIndexMultiData(method, base_start, amount);
            break;
        }
        [[fallthrough]];
    default:
        for (u32 i = 0; i < amount; i++) {
            CallMethod(system, method, base_start[i], methods_pending - i <= 1);
        }
        break;
    }
}

void Maxwell3D::ProcessMacroUpload(u32 data) {
    macro_engine.AddCode(regs.load_mme.instruction_ptr++, data);
}

void Maxwell3D::ProcessMacroBind(u32 data) {
    macro_positions[regs.load_mme.start_address_ptr++] = data;
}

void Maxwell3D::ProcessFirmwareCall4() {
    LOG_DEBUG(HW_GPU, "(STUBBED) called");

    // Firmware call 4 is a blob that changes some registers depending on its parameters.
    // These registers don't affect emulation and so are stubbed by setting 0xd00 to 1.
    regs.shadow_scratch[0] = 1;
}

void Maxwell3D::StampQueryResult(Core::System& system, u64 payload, bool long_query) {
    const GPUVAddr sequence_address{regs.report_semaphore.Address()};
    if (long_query) {
        memory_manager.Write<u64>(sequence_address + sizeof(u64), system.GPU().GetTicks());
        memory_manager.Write<u64>(sequence_address, payload);
    } else {
        memory_manager.Write<u32>(sequence_address, static_cast<u32>(payload));
    }
}

void Maxwell3D::ProcessQueryGet() {
    VideoCommon::QueryPropertiesFlags flags{};
    if (regs.report_semaphore.query.short_query == 0) {
        flags |= VideoCommon::QueryPropertiesFlags::HasTimeout;
    }
    const GPUVAddr sequence_address{regs.report_semaphore.Address()};
    const VideoCommon::QueryType query_type =
        static_cast<VideoCommon::QueryType>(regs.report_semaphore.query.report.Value());
    const u32 payload = regs.report_semaphore.payload;
    const u32 subreport = regs.report_semaphore.query.sub_report;
    switch (regs.report_semaphore.query.operation) {
    case Regs::ReportSemaphore::Operation::Release:
        if (regs.report_semaphore.query.short_query != 0) {
            flags |= VideoCommon::QueryPropertiesFlags::IsAFence;
        }
        rasterizer->Query(sequence_address, query_type, flags, payload, subreport);
        break;
    case Regs::ReportSemaphore::Operation::Acquire:
        // TODO(Blinkhawk): Under this operation, the GPU waits for the CPU to write a value that
        // matches the current payload.
        UNIMPLEMENTED_MSG("Unimplemented query operation ACQUIRE");
        break;
    case Regs::ReportSemaphore::Operation::ReportOnly:
        rasterizer->Query(sequence_address, query_type, flags, payload, subreport);
        break;
    case Regs::ReportSemaphore::Operation::Trap:
        UNIMPLEMENTED_MSG("Unimplemented query operation TRAP");
        break;
    default:
        UNIMPLEMENTED_MSG("Unknown query operation");
        break;
    }
}

void Maxwell3D::ProcessQueryCondition() {
    if (rasterizer->AccelerateConditionalRendering()) {
        execute_on = true;
        return;
    }
    const GPUVAddr condition_address{regs.render_enable.Address()};
    switch (regs.render_enable_override) {
    case Regs::RenderEnable::Override::AlwaysRender:
        execute_on = true;
        break;
    case Regs::RenderEnable::Override::NeverRender:
        execute_on = false;
        break;
    case Regs::RenderEnable::Override::UseRenderEnable: {
        switch (regs.render_enable.mode) {
        case Regs::RenderEnable::Mode::True: {
            execute_on = true;
            break;
        }
        case Regs::RenderEnable::Mode::False: {
            execute_on = false;
            break;
        }
        case Regs::RenderEnable::Mode::Conditional: {
            Regs::ReportSemaphore::Compare cmp;
            memory_manager.ReadBlock(condition_address, &cmp, sizeof(cmp));
            execute_on = cmp.initial_sequence != 0U && cmp.initial_mode != 0U;
            break;
        }
        case Regs::RenderEnable::Mode::IfEqual: {
            Regs::ReportSemaphore::Compare cmp;
            memory_manager.ReadBlock(condition_address, &cmp, sizeof(cmp));
            execute_on = cmp.initial_sequence == cmp.current_sequence &&
                         cmp.initial_mode == cmp.current_mode;
            break;
        }
        case Regs::RenderEnable::Mode::IfNotEqual: {
            Regs::ReportSemaphore::Compare cmp;
            memory_manager.ReadBlock(condition_address, &cmp, sizeof(cmp));
            execute_on = cmp.initial_sequence != cmp.current_sequence ||
                         cmp.initial_mode != cmp.current_mode;
            break;
        }
        default: {
            UNIMPLEMENTED_MSG("Uninplemented Condition Mode!");
            execute_on = true;
            break;
        }
        }
        break;
    }
    }
}

void Maxwell3D::ProcessCounterReset() {
    const auto query_type = [clear_report = regs.clear_report_value]() {
        switch (clear_report) {
        case Tegra::Engines::Maxwell3D::Regs::ClearReport::ZPassPixelCount:
            return VideoCommon::QueryType::ZPassPixelCount64;
        case Tegra::Engines::Maxwell3D::Regs::ClearReport::StreamingPrimitivesSucceeded:
            return VideoCommon::QueryType::StreamingPrimitivesSucceeded;
        case Tegra::Engines::Maxwell3D::Regs::ClearReport::PrimitivesGenerated:
            return VideoCommon::QueryType::PrimitivesGenerated;
        case Tegra::Engines::Maxwell3D::Regs::ClearReport::VtgPrimitivesOut:
            return VideoCommon::QueryType::VtgPrimitivesOut;
        default:
            LOG_DEBUG(HW_GPU, "Unimplemented counter reset={}", clear_report);
            return VideoCommon::QueryType::Payload;
        }
    }();
    rasterizer->ResetCounter(query_type);
}

void Maxwell3D::ProcessSyncPoint() {
    const u32 sync_point = regs.sync_info.sync_point.Value();
    [[maybe_unused]] const u32 cache_flush = regs.sync_info.clean_l2.Value();
    rasterizer->SignalSyncPoint(sync_point);
}

void Maxwell3D::ProcessCBBind(size_t stage_index) {
    // Bind the buffer currently in CB_ADDRESS to the specified index in the desired shader
    // stage.
    const auto& bind_data = regs.bind_groups[stage_index];
    auto& buffer = state.shader_stages[stage_index].const_buffers[bind_data.shader_slot];
    buffer.enabled = bind_data.valid.Value() != 0;
    buffer.address = regs.const_buffer.Address();
    buffer.size = regs.const_buffer.size;

    const bool is_enabled = bind_data.valid.Value() != 0;
    if (!is_enabled) {
        rasterizer->DisableGraphicsUniformBuffer(stage_index, bind_data.shader_slot);
        return;
    }
    const GPUVAddr gpu_addr = regs.const_buffer.Address();
    const u32 size = regs.const_buffer.size;
    rasterizer->BindGraphicsUniformBuffer(stage_index, bind_data.shader_slot, gpu_addr, size);
}

void Maxwell3D::ProcessCBMultiData(const u32* start_base, u32 amount) {
    // Write the input value to the current const buffer at the current position.
    const GPUVAddr buffer_address = regs.const_buffer.Address();
    ASSERT(buffer_address != 0);

    // Don't allow writing past the end of the buffer.
    ASSERT(regs.const_buffer.offset <= regs.const_buffer.size);

    const GPUVAddr address{buffer_address + regs.const_buffer.offset};
    const size_t copy_size = amount * sizeof(u32);
    memory_manager.WriteBlockCached(address, start_base, copy_size);

    // Increment the current buffer position.
    regs.const_buffer.offset += static_cast<u32>(copy_size);
}

void Maxwell3D::ProcessCBData(u32 value) {
    ProcessCBMultiData(&value, 1);
}

void Maxwell3D::ProcessInlineIndexMultiData(u32 method, const u32* start_base, u32 amount) {
    if (amount == 0) {
        return;
    }
    const u32 argument = ProcessShadowRam(method, start_base[amount - 1]);
    ProcessDirtyRegisters(method, argument);
    draw_manager.SetInlineIndexBuffer(*this, method, start_base, amount);
}

Texture::TICEntry Maxwell3D::GetTICEntry(u32 tic_index) const {
    const GPUVAddr tic_address_gpu{regs.tex_header.Address() +
                                   tic_index * sizeof(Texture::TICEntry)};
    Texture::TICEntry tic_entry;
    memory_manager.ReadBlockUnsafe(tic_address_gpu, &tic_entry, sizeof(Texture::TICEntry));
    return tic_entry;
}

Texture::TSCEntry Maxwell3D::GetTSCEntry(u32 tsc_index) const {
    const GPUVAddr tsc_address_gpu{regs.tex_sampler.Address() +
                                   tsc_index * sizeof(Texture::TSCEntry)};
    Texture::TSCEntry tsc_entry;
    memory_manager.ReadBlockUnsafe(tsc_address_gpu, &tsc_entry, sizeof(Texture::TSCEntry));
    return tsc_entry;
}

u32 Maxwell3D::GetRegisterValue(u32 method) const {
    ASSERT(method < Regs::NUM_REGS && "Invalid Maxwell3D register");
    return regs.reg_array[method];
}

void Maxwell3D::SetHLEReplacementAttributeType(u32 bank, u32 offset,
                                               HLEReplacementAttributeType name) {
    const u64 key = (static_cast<u64>(bank) << 32) | offset;
    replace_table.emplace(key, name);
}

} // namespace Tegra::Engines
