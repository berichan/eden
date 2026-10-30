// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <stop_token>
#include <catch2/catch_test_macros.hpp>
#include "common/scope_exit.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/device_memory.h"
#include "core/hle/kernel/k_page_table.h"
#include "core/hle/kernel/k_process.h"
#include "core/memory.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ampere/graphics_header.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/host_translate_info.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/gpu.h"
#include "video_core/memory_manager.h"
#include "video_core/renderer_null/null_rasterizer.h"
#include "video_core/shader_environment.h"
#include "video_core/surface.h"

TEST_CASE("Native graphics capture separates source headers, code and constant snapshots",
          "[native-shader]") {
    using Environment = VideoCommon::GraphicsEnvironment;
    using Header = Shader::Ampere::GraphicsHeader;
    using Engine = Tegra::Engines::Maxwell3D;
    using Program = Engine::Regs::ShaderType;
    const auto layout = Settings::values.memory_layout_mode.GetValue();
    const auto multicore = Settings::values.use_multi_core.GetValue();
    Settings::values.memory_layout_mode = Settings::MemoryLayout::Memory_12Gb;
    Settings::values.use_multi_core = false;
    SCOPE_EXIT {
        Settings::values.memory_layout_mode = layout;
        Settings::values.use_multi_core = multicore;
    };
    Core::System system;
    system.Initialize();
    Kernel::KProcess process{system.Kernel()};
    auto& table = process.GetPageTable().GetImpl();
    table.Resize(42, 12);
    process.GetMemory().SetCurrentPageTable(process);
    constexpr u64 cpu = (1ULL << 41) + 0x1000;
    constexpr u64 size = 3 * Kernel::PageSize;
    process.GetMemory().MapMemoryRegion(table, cpu, size, Core::DramMemoryMap::Base + (10ULL << 30),
                                        Common::MemoryPermission::ReadWrite, false);
    SCOPE_EXIT {
        process.GetMemory().UnmapRegion(table, cpu, size, false);
    };
    Tegra::GPU gpu{system, false, false};
    Null::RasterizerNull rasterizer{gpu};
    Tegra::MaxwellDeviceMemoryManager backing{system.DeviceMemory()};
    backing.BindInterface(&rasterizer);
    const auto asid = backing.RegisterProcess(&process.GetMemory());
    SCOPE_EXIT {
        backing.UnregisterProcess(asid);
    };
    const auto device = backing.Allocate(size);
    REQUIRE(device != 0);
    backing.Map(device, cpu, size, asid, true);
    SCOPE_EXIT {
        backing.Unmap(device, size);
        backing.Free(device, size);
    };
    Tegra::MemoryManager memory{system, backing, 32, 1ULL << 32, 16, 12};
    memory.BindRasterizer(&rasterizer);
    constexpr GPUVAddr base = 0x8000000;
    REQUIRE(memory.Map(base, device, Kernel::PageSize, Tegra::PTEKind::PITCH, false) == base);
    REQUIRE(memory.Map(base + Kernel::PageSize, device + 2 * Kernel::PageSize, Kernel::PageSize,
                       Tegra::PTEKind::PITCH, false) == base + Kernel::PageSize);
    constexpr GPUVAddr header_address = base + Kernel::PageSize - Header::SizeBytes;
    const std::array<u32, 4> exit{0x0000794d, 0, 0x03800000, 0x000fea00};
    const auto write = [&](GPUVAddr address, const auto& value) {
        REQUIRE(memory.WriteBlockChecked(address,
                                         {reinterpret_cast<const u8*>(&value), sizeof(value)}));
    };
    // The source header ends exactly at a page boundary. The following code is
    // physically discontiguous; interpreting the legacy 80-byte header would read zeros.
    write(header_address + Header::SizeBytes, exit);
    for (const auto program : {Program::VertexB, Program::Pixel}) {
        const bool vertex = program == Program::VertexB;
        Header::Words header{};
        header[0] = (vertex ? 1U : 2U) | (4U << 5) | ((vertex ? 1U : 5U) << 10) | (6U << 17);
        write(header_address, header);
        Engine engine{memory};
        engine.regs.viewport_scale_offset_enabled = 1;
        const auto capture = [&](std::stop_token stop = {}) {
            return Environment{engine, memory, program,
                               Environment::NativeProgram{header_address, stop}};
        };
        auto environment = capture();
        REQUIRE(environment.SourceSizeBytes() == 144);
        REQUIRE(environment.StartAddress() == 80);
        REQUIRE(environment.TranslateProgramAddress(0) == 80);
        REQUIRE(environment.ReadInstruction(80) != 0);
        REQUIRE_FALSE(environment.CanBeSerialized());
        REQUIRE_THROWS_AS(environment.ReadTextureType(0), Shader::RuntimeError);
        REQUIRE_THROWS_AS(environment.ReadTexturePixelFormat(0xffffffff), Shader::RuntimeError);
        REQUIRE_THROWS_AS(environment.IsTexturePixelFormatInteger(0), Shader::RuntimeError);
        Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_pool;
        Shader::ObjectPool<Shader::IR::Inst> inst_pool;
        Shader::ObjectPool<Shader::IR::Block> block_pool;
        Shader::Maxwell::Flow::CFG cfg{environment, flow_pool, environment.StartAddress()};
        Shader::HostTranslateInfo host;
        const auto ir =
            Shader::Maxwell::TranslateProgram(inst_pool, block_pool, environment, cfg, host);
        REQUIRE(ir.stage == (vertex ? Shader::Stage::VertexB : Shader::Stage::Fragment));
        const auto original_hash = environment.CalculateHash();
        REQUIRE(environment.ReadViewportTransformState() == 1);
        engine.regs.viewport_scale_offset_enabled = 0;
        auto viewport_disabled = capture();
        REQUIRE(environment.ReadViewportTransformState() == 1);
        REQUIRE(viewport_disabled.ReadViewportTransformState() == 0);
        REQUIRE(viewport_disabled.CalculateHash() != original_hash);
        engine.regs.viewport_scale_offset_enabled = 2;
        REQUIRE_THROWS_AS(capture(), Shader::RuntimeError);
        engine.regs.viewport_scale_offset_enabled = 1;
        header[31] = 0x12345678;
        write(header_address, header);
        REQUIRE(capture().CalculateHash() != original_hash);
        auto& buffers = engine.state.shader_stages[vertex ? 0 : 4].const_buffers;
        buffers[1] = {base + 0x200, 16, true};
        write(base + 0x200, u32{0x12345678});
        auto frozen = capture();
        write(base + 0x200, u32{0xabcdef01});
        REQUIRE(frozen.ReadCbufValue(1, 0) == 0x12345678);
        auto updated = capture();
        REQUIRE(updated.ReadCbufValue(1, 0) == 0xabcdef01);
        REQUIRE(updated.CalculateHash() != frozen.CalculateHash());
        auto rescheduled = exit;
        rescheduled[3] ^= 0x200; // Scheduling delay is not represented in the lowered instruction.
        write(header_address + Header::SizeBytes, rescheduled);
        auto new_source = capture();
        REQUIRE(new_source.ReadInstruction(80) == updated.ReadInstruction(80));
        REQUIRE(new_source.CalculateHash() != updated.CalculateHash());
        write(header_address + Header::SizeBytes, exit);
        buffers[1] = {};
        REQUIRE(frozen.ReadCbufValue(1, 0) == 0x12345678);
        REQUIRE_THROWS_AS(frozen.ReadCbufValue(1, 13), Shader::RuntimeError);
        REQUIRE_THROWS_AS(environment.ReadCbufValue(1, 0), Shader::RuntimeError);
        REQUIRE_THROWS_AS(environment.ReadCbufValue(18, 0), Shader::RuntimeError);
        buffers[0] = {base + 0x200, 3, true};
        auto short_buffer = capture();
        REQUIRE_THROWS_AS(short_buffer.ReadCbufValue(0, 0), Shader::RuntimeError);
        buffers[0].size = 4;
        auto checked = capture();
        REQUIRE(checked.ReadCbufValue(0, 0) == 0xabcdef01);
        REQUIRE_THROWS_AS(checked.ReadCbufValue(0, 1), Shader::RuntimeError);
        REQUIRE_THROWS_AS(checked.ReadCbufValue(0, 0xffffffff), Shader::RuntimeError);
        buffers[0].address = base + 2 * Kernel::PageSize;
        auto unmapped = capture();
        REQUIRE_THROWS_AS(unmapped.ReadCbufValue(0, 0), Shader::RuntimeError);
        std::stop_source cancelled;
        cancelled.request_stop();
        REQUIRE_THROWS_AS(capture(cancelled.get_token()), Shader::RuntimeError);
        REQUIRE_THROWS_AS((Environment{engine, memory, Program::Geometry,
                                       Environment::NativeProgram{header_address, {}}}),
                          Shader::RuntimeError);
        REQUIRE_THROWS_AS((Environment{engine, memory, program,
                                       Environment::NativeProgram{header_address + 1, {}}}),
                          Shader::RuntimeError);
    }
    // Captured instructions and bank1 remain owned; uncaptured reads must not
    // silently observe a new mapping incarnation.
    Engine engine{memory};
    engine.state.shader_stages[4].const_buffers[0] = {base + 0x200, 4, true};
    Environment captured{engine, memory, Program::Pixel,
                         Environment::NativeProgram{header_address, {}}};
    {
        auto lease = memory.AcquireMappingLease();
        REQUIRE(lease.Protects(memory));
        Environment external{engine, memory, Program::Pixel,
                             Environment::NativeProgram{header_address, {}, &lease}};
        REQUIRE(external.ReadCbufValue(0, 0) == 0xabcdef01);
        REQUIRE_THROWS_AS(external.ReadTextureType(0), Shader::RuntimeError);
        Tegra::MemoryManager other{system, backing, 32, 1ULL << 32, 16, 12};
        REQUIRE_FALSE(lease.Protects(other));
        REQUIRE_THROWS_AS((Environment{engine, other, Program::Pixel,
                                       Environment::NativeProgram{header_address, {}, &lease}}),
                          Shader::RuntimeError);
    }
    // Fixed ISA with changing descriptor metadata must not share specialization.
    using Words = Tegra::Control::NativeDescriptorWords;
    Words texture{};
    texture[0] = 8;
    for (u32 i = 0; i < 4; ++i)
        texture[0] |= (2U << (7 + 3 * i)) | ((i + 2) << (19 + 3 * i));
    texture[1] = 0x100000;
    texture[2] = (3U << 21) | 0x21;
    texture[3] = 3U << 28;
    texture[4] = 15U | (1U << 23);
    texture[5] = 7U | (1U << 31);
    texture[7] = 3U << 4;
    Words sampler{};
    sampler[1] = 2U | (2U << 4) | (2U << 6);
    constexpr u32 Handle = (1U << 20) | 2;
    engine.regs.tex_header = {0, static_cast<u32>(base + 0x600), 2};
    engine.regs.tex_sampler = {0, static_cast<u32>(base + 0x800), 1};
    engine.regs.sampler_binding = Engine::Regs::SamplerBinding::Independently;
    const auto update_descriptors = [&] {
        write(base + 0x640, texture);
        write(base + 0x820, sampler);
    };
    update_descriptors();
    const auto specialize = [&] {
        Environment environment{engine, memory, Program::Pixel,
                                Environment::NativeProgram{header_address, {}}};
        REQUIRE(environment.ReadTextureType(Handle) == Shader::TextureType::Color2D);
        (void)environment.ReadTexturePixelFormat(Handle);
        environment.FinalizeNativeResources();
        REQUIRE(environment.NativeTextures().size() == 1);
        REQUIRE(environment.NativeTextures()[0].texture.width == 16);
        REQUIRE(environment.NativeTextures()[0].texture.height == 8);
        REQUIRE_THROWS_AS(environment.ReadTextureType(Handle + 1), Shader::RuntimeError);
        REQUIRE_THROWS_AS(environment.ReadCbufValue(0, 0), Shader::RuntimeError);
        return environment.CalculateHash();
    };
    const u64 original = specialize();
    REQUIRE(specialize() == original);
    const auto specialize_constant = [&] {
        Environment environment{engine, memory, Program::Pixel,
                                Environment::NativeProgram{header_address, {}}};
        REQUIRE(environment.ReadCbufValue(0, 0) != 0);
        environment.FinalizeNativeResources();
        return environment.CalculateHash();
    };
    const auto constant_hash = specialize_constant();
    write(base + 0x200, u32{0x98765432});
    REQUIRE(specialize_constant() != constant_hash);
    write(base + 0x200, u32{0xabcdef01});
    texture[4] ^= 1U << 22;
    update_descriptors();
    REQUIRE(specialize() != original);
    texture[4] ^= 1U << 22;
    sampler[1] |= 1U << 12;
    update_descriptors();
    REQUIRE(specialize() != original);
    sampler[1] &= ~(1U << 12);
    texture[2] |= 1U << 18;
    update_descriptors();
    REQUIRE(specialize() != original);
    {
        auto lease = memory.AcquireMappingLease();
        Environment external{engine, memory, Program::Pixel,
                             Environment::NativeProgram{header_address, {}, &lease}};
        REQUIRE(external.ReadTextureType(Handle) == Shader::TextureType::Color2D);
        REQUIRE(external.NativeTextures()[0].texture.reserved[2] == (1U << 18));
    }
    // The same shader must specialize the native packed float format separately
    // and retain its alpha-one swizzle instead of reinterpreting it as RGBA8.
    const u64 rgba_hash = specialize();
    texture[0] = 0x21;
    for (u32 i = 0; i < 4; ++i)
        texture[0] |= (7U << (7 + 3 * i)) | ((i == 3 ? 7U : i + 2) << (19 + 3 * i));
    update_descriptors();
    {
        Environment floating{engine, memory, Program::Pixel,
                             Environment::NativeProgram{header_address, {}}};
        REQUIRE(floating.ReadTextureType(Handle) == Shader::TextureType::Color2D);
        REQUIRE(static_cast<VideoCore::Surface::PixelFormat>(floating.ReadTexturePixelFormat(
                    Handle)) == VideoCore::Surface::PixelFormat::B10G11R11_FLOAT);
        REQUIRE_FALSE(floating.IsTexturePixelFormatInteger(Handle));
        REQUIRE(floating.NativeTextures()[0].texture.swizzle[3] == 7);
        floating.FinalizeNativeResources();
        REQUIRE(floating.CalculateHash() != rgba_hash);
    }
    texture[0] = 0x40;
    for (u32 i = 0; i < 4; ++i)
        texture[0] |= (2U << (7 + 3 * i)) | ((i + 2) << (19 + 3 * i));
    texture[4] |= 1U << 22;
    update_descriptors();
    {
        Environment astc{engine, memory, Program::Pixel,
                         Environment::NativeProgram{header_address, {}}};
        REQUIRE(astc.ReadTextureType(Handle) == Shader::TextureType::Color2D);
        REQUIRE(static_cast<VideoCore::Surface::PixelFormat>(astc.ReadTexturePixelFormat(Handle)) ==
                VideoCore::Surface::PixelFormat::ASTC_2D_4X4_SRGB);
        REQUIRE_FALSE(astc.IsTexturePixelFormatInteger(Handle));
    }
    memory.Unmap(base + Kernel::PageSize, Kernel::PageSize);
    REQUIRE(captured.ReadInstruction(80) != 0);
    REQUIRE_THROWS_AS(captured.ReadCbufValue(0, 0), Shader::RuntimeError);
    REQUIRE_THROWS_AS((Environment{engine, memory, Program::Pixel,
                                   Environment::NativeProgram{header_address, {}}}),
                      Shader::RuntimeError);
}
