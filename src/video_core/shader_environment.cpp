// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>

#include <ranges>
#include "common/assert.h"
#include "common/cityhash.h"
#include "common/common_types.h"
#include "common/div_ceil.h"
#include "common/fs/fs.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ampere/graphics_header.h"
#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"
#include "shader_recompiler/frontend/ampere/program_address_map.h"
#include "shader_recompiler/frontend/ampere/program_capture.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/object_pool.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/memory_manager.h"
#include "video_core/shader_environment.h"
#include "video_core/texture_cache/format_lookup_table.h"
#include "video_core/textures/texture.h"

namespace VideoCommon {

constexpr std::array<char, 8> MAGIC_NUMBER{'y', 'u', 'z', 'u', 'c', 'a', 'c', 'h'};

constexpr size_t INST_SIZE = sizeof(u64);

using Maxwell = Tegra::Engines::Maxwell3D::Regs;

static u64 MakeCbufKey(u32 index, u32 offset) {
    return (static_cast<u64>(index) << 32) | offset;
}

static Shader::TextureType ConvertTextureType(const Tegra::Texture::TICEntry& entry) {
    switch (entry.texture_type) {
    case Tegra::Texture::TextureType::Texture1D:
        return Shader::TextureType::Color1D;
    case Tegra::Texture::TextureType::Texture2D:
    case Tegra::Texture::TextureType::Texture2DNoMipmap:
        return entry.normalized_coords ? Shader::TextureType::Color2D
                                       : Shader::TextureType::Color2DRect;
    case Tegra::Texture::TextureType::Texture3D:
        return Shader::TextureType::Color3D;
    case Tegra::Texture::TextureType::TextureCubemap:
        return Shader::TextureType::ColorCube;
    case Tegra::Texture::TextureType::Texture1DArray:
        return Shader::TextureType::ColorArray1D;
    case Tegra::Texture::TextureType::Texture2DArray:
        return Shader::TextureType::ColorArray2D;
    case Tegra::Texture::TextureType::Texture1DBuffer:
        return Shader::TextureType::Buffer;
    case Tegra::Texture::TextureType::TextureCubeArray:
        return Shader::TextureType::ColorArrayCube;
    default:
        LOG_ERROR(Shader, "Invalid texture_type={}. Falling back to texture_type={}", static_cast<int>(entry.texture_type.Value()), Shader::TextureType::Color2D);
        return Shader::TextureType::Color2D;
    }
}

static Shader::TexturePixelFormat ConvertTexturePixelFormat(const Tegra::Texture::TICEntry& entry) {
    return static_cast<Shader::TexturePixelFormat>(
        PixelFormatFromTextureInfo(entry.format, entry.r_type, entry.g_type, entry.b_type,
                                   entry.a_type, entry.srgb_conversion));
}

static std::string_view StageToPrefix(Shader::Stage stage) {
    switch (stage) {
    case Shader::Stage::VertexB:
        return "vs";
    case Shader::Stage::TessellationControl:
        return "tc";
    case Shader::Stage::TessellationEval:
        return "te";
    case Shader::Stage::Geometry:
        return "gs";
    case Shader::Stage::Fragment:
        return "fs";
    case Shader::Stage::Compute:
        return "cs";
    case Shader::Stage::VertexA:
        return "va";
    default:
        return "uk";
    }
}

static void DumpImpl(u64 /*pipeline_hash*/, u64 shader_hash, std::span<const u64> code,
                     [[maybe_unused]] u32 read_highest, [[maybe_unused]] u32 read_lowest,
                     u32 initial_offset, Shader::Stage stage) {
    const auto dump_dir{Common::FS::GetEdenPath(Common::FS::EdenPath::DumpDir)};
    if (!Common::FS::CreateDir(dump_dir)) {
        LOG_ERROR(Common_Filesystem, "Failed to create dump directory");
        return;
    }
    const auto prefix = StageToPrefix(stage);
    const auto name{dump_dir /
                    fmt::format("{:016x}_{:016x}_{}.ash",
                                Settings::GetCurrentProgramID(), shader_hash, prefix)};
    std::fstream shader_file(name, std::ios::out | std::ios::binary);
    ASSERT(initial_offset % sizeof(u64) == 0);
    const size_t jump_index = initial_offset / sizeof(u64);
    const size_t code_size = code.size_bytes() - initial_offset;
    shader_file.write(reinterpret_cast<const char*>(&code[jump_index]), code_size);

    // + 1 instruction, due to the fact that we skip the final self branch instruction in the code,
    // but we need to consider it for padding, otherwise nvdisasm rages.
    const size_t padding_needed = (32 - ((code_size + INST_SIZE) % 32)) % 32;
    for (size_t i = 0; i < INST_SIZE + padding_needed; i++) {
        shader_file.put(0);
    }
}

GenericEnvironment::GenericEnvironment(Tegra::MemoryManager& gpu_memory_, GPUVAddr program_base_,
                                       u32 start_address_)
    : gpu_memory{&gpu_memory_}, program_base{program_base_} {
    start_address = start_address_;
}

GenericEnvironment::~GenericEnvironment() = default;

u32 GenericEnvironment::TextureBoundBuffer() const {
    return texture_bound;
}

u32 GenericEnvironment::LocalMemorySize() const {
    return local_memory_size;
}

u32 GenericEnvironment::SharedMemorySize() const {
    return shared_memory_size;
}

std::array<u32, 3> GenericEnvironment::WorkgroupSize() const {
    return workgroup_size;
}

u64 GenericEnvironment::ReadInstruction(u32 address) {
    read_lowest = (std::min)(read_lowest, address);
    read_highest = (std::max)(read_highest, address);

    if (address >= cached_lowest && address < cached_highest) {
        return code[(address - cached_lowest) / INST_SIZE];
    }
    if (native_source_size) {
        throw Shader::RuntimeError("Lowered native instruction outside captured code: {:#x}",
                                   address);
    }
    has_unbound_instructions = true;
    return gpu_memory->Read<u64>(program_base + address);
}

u32 GenericEnvironment::TranslateProgramAddress(u32 address) const {
    if (const auto translated = Shader::Ampere::NativeProgramAddress(code, address)) {
        return *translated;
    }
    if (native_source_size) {
        throw Shader::RuntimeError("Invalid native branch address: {:#x}", address);
    }
    return address;
}

std::optional<u64> GenericEnvironment::Analyze() {
    if (native_source_size) {
        return CalculateHash();
    }
    const std::optional<u64> size{TryFindSize()};
    if (!size) {
        return std::nullopt;
    }
    cached_lowest = start_address;
    cached_highest = start_address + static_cast<u32>(*size);
    return Common::CityHash64(reinterpret_cast<const char*>(code.data()), *size);
}

void GenericEnvironment::SetCachedSize(size_t size_bytes) {
    if (native_source_size) {
        if (size_bytes != CachedSizeBytes()) {
            throw Shader::RuntimeError("Native shader changed during pipeline construction");
        }
        return;
    }
    cached_lowest = start_address;
    cached_highest = start_address + static_cast<u32>(size_bytes);
    code.resize(CachedSizeWords());
    gpu_memory->ReadBlock(program_base + cached_lowest, code.data(), code.size() * sizeof(u64));
}

size_t GenericEnvironment::CachedSizeWords() const noexcept {
    return CachedSizeBytes() / INST_SIZE;
}

size_t GenericEnvironment::CachedSizeBytes() const noexcept {
    return static_cast<size_t>(cached_highest) - cached_lowest + INST_SIZE;
}

size_t GenericEnvironment::ReadSizeBytes() const noexcept {
    return read_highest - read_lowest + INST_SIZE;
}

bool GenericEnvironment::CanBeSerialized() const noexcept {
    // The current disk format cannot retain native source/specialization identity.
    return !native_source_size && !has_unbound_instructions;
}

u64 GenericEnvironment::CalculateHash() const {
    if (native_source_size) {
        constexpr u64 NativeCacheVersion = 0x4f554e4345000004ULL;
        return Common::CityHash64(reinterpret_cast<const char*>(code.data()),
                                  code.size() * sizeof(u64)) ^
               NativeCacheVersion ^ (u64{local_memory_size} << 32) ^ native_specialization_hash ^
               native_resource_hash;
    }
    const size_t size{ReadSizeBytes()};
    const auto data{std::make_unique<char[]>(size)};
    gpu_memory->ReadBlock(program_base + read_lowest, data.get(), size);
    return Common::CityHash64(data.get(), size);
}

void GenericEnvironment::Dump(u64 pipeline_hash, u64 shader_hash) {
    DumpImpl(pipeline_hash, shader_hash, code, read_highest, read_lowest, initial_offset, stage);
}

void GenericEnvironment::Serialize(std::ostream& file) const {
    const u64 code_size{static_cast<u64>(CachedSizeBytes())};
    const u64 num_texture_types{static_cast<u64>(texture_types.size())};
    const u64 num_texture_pixel_formats{static_cast<u64>(texture_pixel_formats.size())};
    const u64 num_cbuf_values{static_cast<u64>(cbuf_values.size())};
    const u64 num_cbuf_replacement_values{static_cast<u64>(cbuf_replacements.size())};

    file.write(reinterpret_cast<const char*>(&code_size), sizeof(code_size))
        .write(reinterpret_cast<const char*>(&num_texture_types), sizeof(num_texture_types))
        .write(reinterpret_cast<const char*>(&num_texture_pixel_formats),
               sizeof(num_texture_pixel_formats))
        .write(reinterpret_cast<const char*>(&num_cbuf_values), sizeof(num_cbuf_values))
        .write(reinterpret_cast<const char*>(&num_cbuf_replacement_values),
               sizeof(num_cbuf_replacement_values))
        .write(reinterpret_cast<const char*>(&local_memory_size), sizeof(local_memory_size))
        .write(reinterpret_cast<const char*>(&texture_bound), sizeof(texture_bound))
        .write(reinterpret_cast<const char*>(&start_address), sizeof(start_address))
        .write(reinterpret_cast<const char*>(&cached_lowest), sizeof(cached_lowest))
        .write(reinterpret_cast<const char*>(&cached_highest), sizeof(cached_highest))
        .write(reinterpret_cast<const char*>(&viewport_transform_state),
               sizeof(viewport_transform_state))
        .write(reinterpret_cast<const char*>(&stage), sizeof(stage))
        .write(reinterpret_cast<const char*>(code.data()), code_size);
    for (const auto& [key, type] : texture_types) {
        file.write(reinterpret_cast<const char*>(&key), sizeof(key))
            .write(reinterpret_cast<const char*>(&type), sizeof(type));
    }
    for (const auto& [key, format] : texture_pixel_formats) {
        file.write(reinterpret_cast<const char*>(&key), sizeof(key))
            .write(reinterpret_cast<const char*>(&format), sizeof(format));
    }
    for (const auto& [key, type] : cbuf_values) {
        file.write(reinterpret_cast<const char*>(&key), sizeof(key))
            .write(reinterpret_cast<const char*>(&type), sizeof(type));
    }
    for (const auto& [key, type] : cbuf_replacements) {
        file.write(reinterpret_cast<const char*>(&key), sizeof(key))
            .write(reinterpret_cast<const char*>(&type), sizeof(type));
    }
    if (stage == Shader::Stage::Compute) {
        file.write(reinterpret_cast<const char*>(&workgroup_size), sizeof(workgroup_size))
            .write(reinterpret_cast<const char*>(&shared_memory_size), sizeof(shared_memory_size));
    } else {
        file.write(reinterpret_cast<const char*>(&sph), sizeof(sph));
        if (stage == Shader::Stage::Geometry) {
            file.write(reinterpret_cast<const char*>(&gp_passthrough_mask),
                       sizeof(gp_passthrough_mask));
        }
    }
}

std::optional<u64> GenericEnvironment::TryFindSize() {
    static constexpr size_t BLOCK_SIZE = 0x1000;
    static constexpr size_t MAXIMUM_SIZE = 0x100000;

    static constexpr u64 SELF_BRANCH_A = 0xE2400FFFFF87000FULL;
    static constexpr u64 SELF_BRANCH_B = 0xE2400FFFFF07000FULL;

    static constexpr u64 EXIT_VALUE = 0xE30000000007000FULL;

    code.resize(MAXIMUM_SIZE / INST_SIZE);

    GPUVAddr guest_addr{program_base + start_address};
    size_t offset{0};
    size_t size{BLOCK_SIZE};
    while (size <= MAXIMUM_SIZE) {
        u64* const data = code.data() + offset / INST_SIZE;
        gpu_memory->ReadBlock(guest_addr, data, BLOCK_SIZE);
        for (size_t index = 0; index < BLOCK_SIZE; index += INST_SIZE) {
            const u64 inst = data[index / INST_SIZE];
            if (inst == SELF_BRANCH_A || inst == SELF_BRANCH_B) {
                return offset + index;
            }
            if (!is_proprietary_driver && inst == EXIT_VALUE) {
                return offset + index + INST_SIZE;
            }
        }
        guest_addr += BLOCK_SIZE;
        size += BLOCK_SIZE;
        offset += BLOCK_SIZE;
    }
    return std::nullopt;
}

void GenericEnvironment::InitializeNativeCode(u32 source_header_size, std::stop_token stop,
                                              const Tegra::MemoryManager::MappingLease* mapping) {
    using Shader::Ampere::Instruction;
    if (!gpu_memory->IsWithinGPUAddressRange(program_base)) {
        throw Shader::RuntimeError("Invalid native shader base: {:#x}", program_base);
    }
    Shader::Ampere::ProgramCaptureState capture;
    std::vector<Instruction> native;
    bool complete{};
    for (u32 index = 0; index < capture.MaximumInstructions; ++index) {
        if (stop.stop_requested())
            throw Shader::RuntimeError("Native shader capture cancelled");
        const GPUVAddr address =
            program_base + source_header_size + u64{index} * sizeof(Instruction);
        Instruction instruction;
        const std::span output{reinterpret_cast<u8*>(&instruction), sizeof(instruction)};
        if (!(mapping ? mapping->ReadBlockChecked(address, output)
                      : gpu_memory->ReadBlockChecked(address, output))) {
            throw Shader::RuntimeError("Unmapped native shader instruction: {:#x}", address);
        }
        native.push_back(instruction);
        const auto status = capture.Observe(instruction, index);
        if (status == Shader::Ampere::ProgramCaptureState::Result::InvalidTarget) {
            throw Shader::RuntimeError("Native shader branch leaves capture bounds");
        }
        if (status == Shader::Ampere::ProgramCaptureState::Result::Complete) {
            complete = true;
            break;
        }
    }
    if (!complete) {
        std::array<u32, 4> counts{};
        for (u32 index = 0; index < native.size(); ++index) {
            const auto& instruction = native[index];
            const u32 opcode = instruction.Bits(0, 12);
            const u32 kind = opcode == 0x94d ? 0 : opcode == 0x950 ? 1 : opcode == 0x944 ? 2 : 3;
            if (kind == 3 && opcode != 0x947 && opcode != 0x945)
                continue;
            if (counts[kind]++ < 4) {
                LOG_DEBUG(HW_GPU,
                          "Native capture control: address={:#x}, index={}, opcode={:#x}, "
                          "predicate={:#x}, displacement={:#x}/{:#x}",
                          program_base, index, opcode, instruction.Bits(12, 4),
                          instruction.words[1],
                          instruction.words[2] & (opcode == 0x945 ? 0 : 0x3ffffU));
            }
        }
        LOG_DEBUG(HW_GPU,
                  "Native capture limit: address={:#x}, instructions={}, exits={}, returns={}, "
                  "calls={}, branches={}",
                  program_base, native.size(), counts[0], counts[1], counts[2], counts[3]);
        throw Shader::RuntimeError("Native shader exceeds capture limit");
    }
    const auto lowered = Shader::Ampere::LowerToMaxwell(native);
    if (stop.stop_requested())
        throw Shader::RuntimeError("Native shader capture cancelled");
    if (!lowered) {
        const u32 failed = lowered.error_offset / sizeof(Instruction);
        const u32 first = failed > 2 ? failed - 2 : 0;
        const u32 last = static_cast<u32>((std::min)(u64{native.size()}, u64{failed} + 4));
        for (u32 index = first; index < last; ++index) {
            const auto& instruction = native[index];
            LOG_DEBUG(HW_GPU,
                      "Native lowering context: address={:#x}, index={}, failed={}, "
                      "words={:#x}/{:#x}/{:#x}/{:#x}",
                      program_base, index, index == failed, instruction.words[0],
                      instruction.words[1], instruction.words[2], instruction.words[3]);
        }
        throw Shader::RuntimeError("Native shader at {:#x}, instruction {:#x}: {}", program_base,
                                   lowered.error_offset, lowered.error);
    }
    native_specialization_hash =
        Common::CityHash64WithSeed(reinterpret_cast<const char*>(native.data()),
                                   native.size() * sizeof(Instruction), native_specialization_hash);
    if (lowered.required_texture_buffer != ~0U) {
        texture_bound = lowered.required_texture_buffer;
    }
    code.resize(sizeof(sph) / sizeof(u64));
    std::memcpy(code.data(), &sph, sizeof(sph));
    for (u64 instruction : lowered.instructions) {
        if ((code.size() * sizeof(u64)) % 32 == 0) {
            code.push_back(0);
        }
        code.push_back(instruction);
    }
    code.push_back(0);
    Shader::Ampere::AppendNativeAddressMap(code, lowered.source_instruction_offsets);
    native_source_size = source_header_size + lowered.source_size;
    cached_lowest = 0;
    cached_highest = static_cast<u32>((code.size() - 1) * sizeof(u64));
}

Tegra::Texture::TICEntry GenericEnvironment::ReadTextureInfo(GPUVAddr tic_addr, u32 tic_limit,
                                                             bool via_header_index, u32 raw) {
    if (native_source_size) {
        throw Shader::RuntimeError("Native texture descriptors require native decoding");
    }
    const auto handle{Tegra::Texture::TexturePair(raw, via_header_index)};
    ASSERT(handle.first <= tic_limit);
    const GPUVAddr descriptor_addr{tic_addr + handle.first * sizeof(Tegra::Texture::TICEntry)};
    Tegra::Texture::TICEntry entry;
    gpu_memory->ReadBlock(descriptor_addr, &entry, sizeof(entry));
    return entry;
}

GraphicsEnvironment::GraphicsEnvironment(Tegra::Engines::Maxwell3D& maxwell3d_,
                                         Tegra::MemoryManager& gpu_memory_,
                                         Maxwell::ShaderType program, GPUVAddr program_base_,
                                         u32 start_address_)
    : GenericEnvironment{gpu_memory_, program_base_, start_address_}, maxwell3d{&maxwell3d_} {
    gpu_memory->ReadBlock(program_base + start_address, &sph, sizeof(sph));
    initial_offset = sizeof(sph);
    gp_passthrough_mask = maxwell3d->regs.post_vtg_shader_attrib_skip_mask;
    switch (program) {
    case Maxwell::ShaderType::VertexA:
        stage = Shader::Stage::VertexA;
        stage_index = 0;
        break;
    case Maxwell::ShaderType::VertexB:
        stage = Shader::Stage::VertexB;
        stage_index = 0;
        break;
    case Maxwell::ShaderType::TessellationInit:
        stage = Shader::Stage::TessellationControl;
        stage_index = 1;
        break;
    case Maxwell::ShaderType::Tessellation:
        stage = Shader::Stage::TessellationEval;
        stage_index = 2;
        break;
    case Maxwell::ShaderType::Geometry:
        stage = Shader::Stage::Geometry;
        stage_index = 3;
        break;
    case Maxwell::ShaderType::Pixel:
        stage = Shader::Stage::Fragment;
        stage_index = 4;
        break;
    default:
        ASSERT_MSG(false, "Invalid program={}", program);
        break;
    }
    const u64 local_size{sph.LocalMemorySize()};
    ASSERT(local_size <= (std::numeric_limits<u32>::max)());
    local_memory_size = static_cast<u32>(local_size) + sph.common3.shader_local_memory_crs_size;
    texture_bound = maxwell3d->regs.bindless_texture_const_buffer_slot;
    is_proprietary_driver = texture_bound == 2;
    has_hle_engine_state =
        maxwell3d->engine_state == Tegra::Engines::Maxwell3D::EngineHint::OnHLEMacro;
}

GraphicsEnvironment::GraphicsEnvironment(Tegra::Engines::Maxwell3D& maxwell3d_,
                                         Tegra::MemoryManager& gpu_memory_,
                                         Maxwell::ShaderType program, NativeProgram native)
    : GenericEnvironment{gpu_memory_, native.header_address, 0}, maxwell3d{&maxwell3d_} {
    using Header = Shader::Ampere::GraphicsHeader;
    if (native.stop.stop_requested())
        throw Shader::RuntimeError("Native graphics capture cancelled");
    if (program != Maxwell::ShaderType::VertexB && program != Maxwell::ShaderType::Pixel)
        throw Shader::RuntimeError("Unsupported native graphics stage: {}",
                                   static_cast<u32>(program));
    constexpr u64 AddressLimit = u64{1} << 40;
    if (program_base >= AddressLimit || Header::SizeBytes > AddressLimit - program_base ||
        (program_base & 15))
        throw Shader::RuntimeError("Invalid native graphics header address: {:#x}", program_base);
    std::optional<Tegra::MemoryManager::MappingLease> owned_mapping;
    if (!native.mapping)
        owned_mapping.emplace(gpu_memory->AcquireMappingLease());
    const auto& mapping = native.mapping ? *native.mapping : *owned_mapping;
    if (!mapping.Protects(*gpu_memory))
        throw Shader::RuntimeError(
            "Native graphics capture lease does not protect its address space");
    native_mapping_generation = gpu_memory->MappingGeneration();
    native_capture_mapping = &mapping;
    SCOPE_EXIT {
        if (!native.mapping)
            native_capture_mapping = nullptr;
    };
    Header::Words source{};
    if (!mapping.ReadBlockChecked(program_base,
                                  {reinterpret_cast<u8*>(source.data()), sizeof(source)}))
        throw Shader::RuntimeError("Unmapped native graphics header: {:#x}", program_base);
    const bool vertex = program == Maxwell::ShaderType::VertexB;
    const auto decoded =
        Header::Decode(source, vertex ? Header::Stage::Vertex : Header::Stage::Fragment);
    if (!decoded.header)
        throw Shader::RuntimeError("Unsupported native graphics header: error={}, words={:#x}",
                                   static_cast<u32>(decoded.error), decoded.unsupported_words);
    sph = decoded.header->metadata;
    local_memory_size = decoded.header->local_memory_size;
    stage = vertex ? Shader::Stage::VertexB : Shader::Stage::Fragment;
    stage_index = vertex ? 0 : 4;
    native_constant_buffers = maxwell3d->state.shader_stages[stage_index].const_buffers;
    initial_offset = sizeof(sph);
    is_proprietary_driver = true;
    texture_bound = maxwell3d->regs.bindless_texture_const_buffer_slot;
    viewport_transform_state = maxwell3d->GetRegisterValue(0x64b);
    if (viewport_transform_state > 1)
        throw Shader::RuntimeError("Invalid native viewport transform state");
    native_specialization_hash =
        Common::CityHash64(reinterpret_cast<const char*>(source.data()), sizeof(source));
    native_specialization_hash =
        Common::CityHash64WithSeed(reinterpret_cast<const char*>(&viewport_transform_state),
                                   sizeof(viewport_transform_state), native_specialization_hash);
    const auto& driver = native_constant_buffers[1];
    if (driver.enabled) {
        if (driver.size == 0 || driver.size > 0x1ffff || driver.address >= AddressLimit ||
            driver.size > AddressLimit - driver.address)
            throw Shader::RuntimeError("Invalid native graphics driver constant range");
        native_driver_constants.resize(driver.size);
        if (!mapping.ReadBlockChecked(driver.address, native_driver_constants))
            throw Shader::RuntimeError("Unmapped native graphics driver constants");
        native_specialization_hash = Common::CityHash64WithSeed(
            reinterpret_cast<const char*>(native_driver_constants.data()),
            native_driver_constants.size(), native_specialization_hash);
    }
    InitializeNativeCode(Header::SizeBytes, native.stop, &mapping);
    start_address = sizeof(sph);
    Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_pool;
    Shader::Maxwell::Flow::CFG cfg{*this, flow_pool, start_address};
    std::vector<std::pair<u64, u32>> dependencies{cbuf_values.begin(), cbuf_values.end()};
    std::ranges::sort(dependencies);
    for (const auto& [key, value] : dependencies) {
        const std::array<u64, 2> dependency{key, value};
        native_specialization_hash =
            Common::CityHash64WithSeed(reinterpret_cast<const char*>(dependency.data()),
                                       sizeof(dependency), native_specialization_hash);
    }
    if (native.stop.stop_requested())
        throw Shader::RuntimeError("Native graphics capture cancelled");
}

u32 GraphicsEnvironment::ReadCbufValue(u32 cbuf_index, u32 cbuf_offset) {
    if (native_source_size) {
        const auto& buffers = native_constant_buffers;
        if (cbuf_index >= buffers.size())
            throw Shader::RuntimeError("Invalid native graphics constant buffer index");
        const auto& buffer = buffers[cbuf_index];
        if (!buffer.enabled || cbuf_offset > buffer.size ||
            sizeof(u32) > buffer.size - cbuf_offset || buffer.address >= (u64{1} << 40) ||
            buffer.size > (u64{1} << 40) - buffer.address)
            throw Shader::RuntimeError("Invalid native graphics constant buffer range");
        const auto key = MakeCbufKey(cbuf_index, cbuf_offset);
        if (const auto cached = cbuf_values.find(key); cached != cbuf_values.end())
            return cached->second;
        if (native_resources_finalized)
            throw Shader::RuntimeError("Native constant dependency changed after preparation");
        u32 value{};
        if (cbuf_index == 1) {
            if (cbuf_offset > native_driver_constants.size() ||
                sizeof(value) > native_driver_constants.size() - cbuf_offset)
                throw Shader::RuntimeError("Invalid native graphics driver constant read");
            std::memcpy(&value, native_driver_constants.data() + cbuf_offset, sizeof(value));
        } else {
            const std::span output{reinterpret_cast<u8*>(&value), sizeof(value)};
            if (native_mapping_generation != gpu_memory->MappingGeneration() ||
                !(native_capture_mapping
                      ? native_capture_mapping->ReadBlockChecked(buffer.address + cbuf_offset,
                                                                 output)
                      : gpu_memory->ReadBlockChecked(buffer.address + cbuf_offset, output)) ||
                native_mapping_generation != gpu_memory->MappingGeneration())
                throw Shader::RuntimeError("Unmapped or changed native graphics constant buffer");
        }
        cbuf_values.emplace(key, value);
        return value;
    }
    const auto& cbuf{maxwell3d->state.shader_stages[stage_index].const_buffers[cbuf_index]};
    ASSERT(cbuf.enabled);
    u32 value{};
    if (cbuf_offset < cbuf.size) {
        value = gpu_memory->Read<u32>(cbuf.address + cbuf_offset);
    }
    cbuf_values.emplace(MakeCbufKey(cbuf_index, cbuf_offset), value);
    return value;
}

std::optional<Shader::ReplaceConstant> GraphicsEnvironment::GetReplaceConstBuffer(u32 bank,
                                                                                  u32 offset) {
    if (!has_hle_engine_state) {
        return std::nullopt;
    }
    const u64 key = (static_cast<u64>(bank) << 32) | static_cast<u64>(offset);
    auto it = maxwell3d->replace_table.find(key);
    if (it == maxwell3d->replace_table.end()) {
        return std::nullopt;
    }
    const auto converted_value = [](Tegra::Engines::Maxwell3D::HLEReplacementAttributeType name) {
        switch (name) {
        case Tegra::Engines::Maxwell3D::HLEReplacementAttributeType::BaseVertex:
            return Shader::ReplaceConstant::BaseVertex;
        case Tegra::Engines::Maxwell3D::HLEReplacementAttributeType::BaseInstance:
            return Shader::ReplaceConstant::BaseInstance;
        case Tegra::Engines::Maxwell3D::HLEReplacementAttributeType::DrawID:
            return Shader::ReplaceConstant::DrawID;
        default:
            UNREACHABLE();
        }
    }(it->second);
    cbuf_replacements.emplace(key, converted_value);
    return converted_value;
}

Shader::TextureType GraphicsEnvironment::ReadTextureType(u32 handle) {
    if (native_source_size) {
        const auto& texture = CaptureNativeTexture(handle).texture;
        return texture.normalized ? Shader::TextureType::Color2D : Shader::TextureType::Color2DRect;
    }
    const auto& regs{maxwell3d->regs};
    const bool via_header_index{regs.sampler_binding == Maxwell::SamplerBinding::ViaHeaderBinding};
    auto entry =
        ReadTextureInfo(regs.tex_header.Address(), regs.tex_header.limit, via_header_index, handle);
    const Shader::TextureType result{ConvertTextureType(entry)};
    texture_types.emplace(handle, result);
    return result;
}

Shader::TexturePixelFormat GraphicsEnvironment::ReadTexturePixelFormat(u32 handle) {
    if (native_source_size) {
        const auto& texture = CaptureNativeTexture(handle).texture;
        if (texture.IsB10G11R11Float2D())
            return static_cast<Shader::TexturePixelFormat>(
                VideoCore::Surface::PixelFormat::B10G11R11_FLOAT);
        if (texture.IsAstc4x4Unorm2D())
            return static_cast<Shader::TexturePixelFormat>(
                texture.srgb ? VideoCore::Surface::PixelFormat::ASTC_2D_4X4_SRGB
                             : VideoCore::Surface::PixelFormat::ASTC_2D_4X4_UNORM);
        return static_cast<Shader::TexturePixelFormat>(
            texture.srgb ? VideoCore::Surface::PixelFormat::A8B8G8R8_SRGB
                         : VideoCore::Surface::PixelFormat::A8B8G8R8_UNORM);
    }
    const auto& regs{maxwell3d->regs};
    const bool via_header_index{regs.sampler_binding == Maxwell::SamplerBinding::ViaHeaderBinding};
    auto entry =
        ReadTextureInfo(regs.tex_header.Address(), regs.tex_header.limit, via_header_index, handle);
    const Shader::TexturePixelFormat result(ConvertTexturePixelFormat(entry));
    texture_pixel_formats.emplace(handle, result);
    return result;
}

bool GraphicsEnvironment::IsTexturePixelFormatInteger(u32 handle) {
    return VideoCore::Surface::IsPixelFormatInteger(
        static_cast<VideoCore::Surface::PixelFormat>(ReadTexturePixelFormat(handle)));
}

const GraphicsEnvironment::NativeTextureBinding& GraphicsEnvironment::CaptureNativeTexture(
    u32 handle) {
    using namespace Tegra::Control;
    for (const auto& binding : native_textures)
        if (binding.handle == handle)
            return binding;
    if (native_resources_finalized || native_textures.size() >= 64)
        throw Shader::RuntimeError("Native texture dependency changed or exceeds capture capacity");
    const auto read = [&](u32 method) { return maxwell3d->GetRegisterValue(method); };
    if (read(0x48d) != 0)
        throw Shader::RuntimeError("Unsupported native linked sampler binding");
    std::optional<Tegra::MemoryManager::MappingLease> owned_mapping;
    if (!native_capture_mapping)
        owned_mapping.emplace(gpu_memory->AcquireMappingLease());
    const auto& mapping = native_capture_mapping ? *native_capture_mapping : *owned_mapping;
    if (!mapping.Protects(*gpu_memory) ||
        native_mapping_generation != gpu_memory->MappingGeneration())
        throw Shader::RuntimeError("Native texture capture address space changed");
    const auto capture = [&](NativeDescriptorPool pool, u32 base, u32 index) {
        const auto address =
            NativeDescriptorAddress(pool, read(base), read(base + 1), read(base + 2), index);
        NativeDescriptorWords words{};
        if (!address || !mapping.ReadBlockChecked(
                            *address, {reinterpret_cast<u8*>(words.data()), sizeof(words)}))
            throw Shader::RuntimeError("Unmapped or invalid native descriptor pool entry");
        return words;
    };
    const auto texture_words = capture(NativeDescriptorPool::Texture, 0x55d, handle & 0xfffff);
    const auto sampler_words = capture(NativeDescriptorPool::Sampler, 0x557, handle >> 20);
    const auto texture = NativeTextureDescriptor::Decode(texture_words);
    const auto sampler = NativeSamplerDescriptor::Decode(sampler_words);
    if (!texture || !sampler ||
        !(texture->IsRgba8Unorm2D() || texture->IsB10G11R11Float2D() ||
          texture->IsAstc4x4Unorm2D())) {
        LOG_DEBUG(HW_GPU,
                  "Native texture metadata rejected: handle={:#x}, "
                  "tic={:#x}/{:#x}/{:#x}/{:#x}/{:#x}/{:#x}/{:#x}/{:#x}, "
                  "tsc={:#x}/{:#x}/{:#x}/{:#x}/{:#x}/{:#x}/{:#x}/{:#x}",
                  handle, texture_words[0], texture_words[1], texture_words[2], texture_words[3],
                  texture_words[4], texture_words[5], texture_words[6], texture_words[7],
                  sampler_words[0], sampler_words[1], sampler_words[2], sampler_words[3],
                  sampler_words[4], sampler_words[5], sampler_words[6], sampler_words[7]);
        throw Shader::RuntimeError("Unsupported native texture or sampler metadata");
    }
    LOG_DEBUG(HW_GPU,
              "Native texture metadata: stage={}, handle={:#x}, address={:#x}, extent={}x{}, "
              "mips={}..{}, reserved_word2={:#x}, sampler_lod={}..{}",
              stage_index, handle, texture->address, texture->width, texture->height,
              texture->view_min_mip, texture->view_max_mip, texture->reserved[2], sampler->min_lod,
              sampler->max_lod);
    native_textures.push_back({handle, *texture, *sampler});
    return native_textures.back();
}

void GraphicsEnvironment::FinalizeNativeResources() {
    if (!native_source_size)
        throw Shader::RuntimeError("Native resource preparation requires a native program");
    if (native_resources_finalized)
        return;
    std::vector<std::pair<u64, u32>> dependencies{cbuf_values.begin(), cbuf_values.end()};
    std::ranges::sort(dependencies);
    u64 hash{};
    for (const auto& [key, value] : dependencies) {
        const std::array<u64, 2> dependency{key, value};
        hash = Common::CityHash64WithSeed(reinterpret_cast<const char*>(dependency.data()),
                                          sizeof(dependency), hash);
    }
    std::ranges::sort(native_textures, {}, &NativeTextureBinding::handle);
    for (const auto& binding : native_textures) {
        hash = Common::CityHash64WithSeed(reinterpret_cast<const char*>(&binding.handle),
                                          sizeof(binding.handle), hash);
        hash =
            Common::CityHash64WithSeed(reinterpret_cast<const char*>(binding.texture.words.data()),
                                       sizeof(binding.texture.words), hash);
        hash =
            Common::CityHash64WithSeed(reinterpret_cast<const char*>(binding.sampler.words.data()),
                                       sizeof(binding.sampler.words), hash);
    }
    native_resource_hash = hash;
    native_resources_finalized = true;
}

u32 GraphicsEnvironment::ReadViewportTransformState() {
    if (native_source_size)
        return viewport_transform_state;
    const auto& regs{maxwell3d->regs};
    viewport_transform_state = regs.viewport_scale_offset_enabled;
    return viewport_transform_state;
}

ComputeEnvironment::ComputeEnvironment(Tegra::Engines::KeplerCompute& kepler_compute_,
                                       Tegra::MemoryManager& gpu_memory_, GPUVAddr program_base_,
                                       u32 start_address_)
    : GenericEnvironment{gpu_memory_, program_base_, start_address_}, kepler_compute{
                                                                          &kepler_compute_} {
    const auto& qmd{kepler_compute->launch_description};
    stage = Shader::Stage::Compute;
    local_memory_size = qmd.local_pos_alloc + qmd.local_crs_alloc;
    texture_bound = kepler_compute->regs.tex_cb_index;
    is_proprietary_driver = texture_bound == 2;
    shared_memory_size = qmd.shared_alloc;
    workgroup_size = {qmd.block_dim_x, qmd.block_dim_y, qmd.block_dim_z};
    if (kepler_compute->IsNativeMode()) {
        is_proprietary_driver = true;
        initial_offset = sizeof(sph);
        InitializeNativeCode(0, kepler_compute->GetNativeStopToken());
        start_address = sizeof(sph);
        // The frontend folds the proprietary driver's constant buffer and resolves indirect
        // branch tables during compilation. Include those inputs in pipeline identity.
        if (qmd.const_buffer_enable_mask.Value() & 2) {
            native_driver_constants.resize(qmd.const_buffer_config[1].size);
            if (!kepler_compute->ReadNativeConstantBuffer(1, 0, native_driver_constants)) {
                throw Shader::RuntimeError("Unmapped native driver constants");
            }
            if (!native_driver_constants.empty()) {
                native_specialization_hash = Common::CityHash64WithSeed(
                    reinterpret_cast<const char*>(native_driver_constants.data()),
                    native_driver_constants.size(), native_specialization_hash);
            }
        }
        Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_pool;
        Shader::Maxwell::Flow::CFG cfg{*this, flow_pool, start_address};
        std::vector<std::pair<u64, u32>> dependencies{cbuf_values.begin(), cbuf_values.end()};
        std::ranges::sort(dependencies);
        for (const auto& [key, value] : dependencies) {
            const std::array<u64, 2> dependency{key, value};
            native_specialization_hash =
                Common::CityHash64WithSeed(reinterpret_cast<const char*>(dependency.data()),
                                           sizeof(dependency), native_specialization_hash);
        }
    }
}

u32 ComputeEnvironment::ReadCbufValue(u32 cbuf_index, u32 cbuf_offset) {
    const auto& qmd{kepler_compute->launch_description};
    if (native_source_size && (cbuf_index >= qmd.const_buffer_config.size() ||
                               !(qmd.const_buffer_enable_mask.Value() & (1U << cbuf_index)))) {
        throw Shader::RuntimeError("Invalid native constant buffer index: {}", cbuf_index);
    }
    ASSERT(((qmd.const_buffer_enable_mask.Value() >> cbuf_index) & 1) != 0);
    const auto& cbuf{qmd.const_buffer_config[cbuf_index]};
    u32 value{};
    if (cbuf_offset < cbuf.size) {
        if (native_source_size) {
            const auto key = MakeCbufKey(cbuf_index, cbuf_offset);
            if (const auto cached = cbuf_values.find(key); cached != cbuf_values.end()) {
                return cached->second;
            }
            if (cbuf_index == 1) {
                if (cbuf_offset > native_driver_constants.size() ||
                    sizeof(value) > native_driver_constants.size() - cbuf_offset) {
                    throw Shader::RuntimeError("Invalid native driver constant read");
                }
                std::memcpy(&value, native_driver_constants.data() + cbuf_offset, sizeof(value));
            } else if (!kepler_compute->ReadNativeConstantBuffer(
                           cbuf_index, cbuf_offset,
                           {reinterpret_cast<u8*>(&value), sizeof(value)})) {
                throw Shader::RuntimeError("Invalid native constant buffer read");
            }
            cbuf_values.emplace(MakeCbufKey(cbuf_index, cbuf_offset), value);
            return value;
        }
        value = gpu_memory->Read<u32>(cbuf.Address() + cbuf_offset);
    }
    cbuf_values.emplace(MakeCbufKey(cbuf_index, cbuf_offset), value);
    return value;
}

Shader::TextureType ComputeEnvironment::ReadTextureType(u32 handle) {
    const auto& regs{kepler_compute->regs};
    const auto& qmd{kepler_compute->launch_description};
    auto entry = ReadTextureInfo(regs.tic.Address(), regs.tic.limit, qmd.linked_tsc != 0, handle);
    const Shader::TextureType result{ConvertTextureType(entry)};
    texture_types.emplace(handle, result);
    return result;
}

Shader::TexturePixelFormat ComputeEnvironment::ReadTexturePixelFormat(u32 handle) {
    const auto& regs{kepler_compute->regs};
    const auto& qmd{kepler_compute->launch_description};
    auto entry = ReadTextureInfo(regs.tic.Address(), regs.tic.limit, qmd.linked_tsc != 0, handle);
    const Shader::TexturePixelFormat result(ConvertTexturePixelFormat(entry));
    texture_pixel_formats.emplace(handle, result);
    return result;
}

bool ComputeEnvironment::IsTexturePixelFormatInteger(u32 handle) {
    return VideoCore::Surface::IsPixelFormatInteger(
        static_cast<VideoCore::Surface::PixelFormat>(ReadTexturePixelFormat(handle)));
}

u32 ComputeEnvironment::ReadViewportTransformState() {
    return viewport_transform_state;
}

void FileEnvironment::Deserialize(std::istream& file) {
    u64 code_size{};
    u64 num_texture_types{};
    u64 num_texture_pixel_formats{};
    u64 num_cbuf_values{};
    u64 num_cbuf_replacement_values{};
    file.read(reinterpret_cast<char*>(&code_size), sizeof(code_size))
        .read(reinterpret_cast<char*>(&num_texture_types), sizeof(num_texture_types))
        .read(reinterpret_cast<char*>(&num_texture_pixel_formats),
              sizeof(num_texture_pixel_formats))
        .read(reinterpret_cast<char*>(&num_cbuf_values), sizeof(num_cbuf_values))
        .read(reinterpret_cast<char*>(&num_cbuf_replacement_values),
              sizeof(num_cbuf_replacement_values))
        .read(reinterpret_cast<char*>(&local_memory_size), sizeof(local_memory_size))
        .read(reinterpret_cast<char*>(&texture_bound), sizeof(texture_bound))
        .read(reinterpret_cast<char*>(&start_address), sizeof(start_address))
        .read(reinterpret_cast<char*>(&read_lowest), sizeof(read_lowest))
        .read(reinterpret_cast<char*>(&read_highest), sizeof(read_highest))
        .read(reinterpret_cast<char*>(&viewport_transform_state), sizeof(viewport_transform_state))
        .read(reinterpret_cast<char*>(&stage), sizeof(stage));
    code.resize(Common::DivCeil(code_size, sizeof(u64)));
    file.read(reinterpret_cast<char*>(code.data()), code_size);
    for (size_t i = 0; i < num_texture_types; ++i) {
        u32 key;
        Shader::TextureType type;
        file.read(reinterpret_cast<char*>(&key), sizeof(key))
            .read(reinterpret_cast<char*>(&type), sizeof(type));
        texture_types.emplace(key, type);
    }
    for (size_t i = 0; i < num_texture_pixel_formats; ++i) {
        u32 key;
        Shader::TexturePixelFormat format;
        file.read(reinterpret_cast<char*>(&key), sizeof(key))
            .read(reinterpret_cast<char*>(&format), sizeof(format));
        texture_pixel_formats.emplace(key, format);
    }
    for (size_t i = 0; i < num_cbuf_values; ++i) {
        u64 key;
        u32 value;
        file.read(reinterpret_cast<char*>(&key), sizeof(key))
            .read(reinterpret_cast<char*>(&value), sizeof(value));
        cbuf_values.emplace(key, value);
    }
    for (size_t i = 0; i < num_cbuf_replacement_values; ++i) {
        u64 key;
        Shader::ReplaceConstant value;
        file.read(reinterpret_cast<char*>(&key), sizeof(key))
            .read(reinterpret_cast<char*>(&value), sizeof(value));
        cbuf_replacements.emplace(key, value);
    }
    if (stage == Shader::Stage::Compute) {
        file.read(reinterpret_cast<char*>(&workgroup_size), sizeof(workgroup_size))
            .read(reinterpret_cast<char*>(&shared_memory_size), sizeof(shared_memory_size));
        initial_offset = Shader::Ampere::HasNativeAddressMap(code) ? sizeof(sph) : 0;
    } else {
        file.read(reinterpret_cast<char*>(&sph), sizeof(sph));
        initial_offset = sizeof(sph);
        if (stage == Shader::Stage::Geometry) {
            file.read(reinterpret_cast<char*>(&gp_passthrough_mask), sizeof(gp_passthrough_mask));
        }
    }
    is_proprietary_driver = texture_bound == 2 || Shader::Ampere::HasNativeAddressMap(code);
}

void FileEnvironment::Dump(u64 pipeline_hash, u64 shader_hash) {
    DumpImpl(pipeline_hash, shader_hash, code, read_highest, read_lowest, initial_offset, stage);
}

u32 FileEnvironment::TranslateProgramAddress(u32 address) const {
    if (const auto translated = Shader::Ampere::NativeProgramAddress(code, address)) {
        return *translated;
    }
    if (Shader::Ampere::HasNativeAddressMap(code)) {
        throw Shader::RuntimeError("Invalid cached native branch address: {:#x}", address);
    }
    return address;
}

u64 FileEnvironment::ReadInstruction(u32 address) {
    if (address < read_lowest || address > read_highest) {
        throw Shader::LogicError("Out of bounds address {}", address);
    }
    return code[(address - read_lowest) / sizeof(u64)];
}

u32 FileEnvironment::ReadCbufValue(u32 cbuf_index, u32 cbuf_offset) {
    const auto it{cbuf_values.find(MakeCbufKey(cbuf_index, cbuf_offset))};
    if (it == cbuf_values.end()) {
        throw Shader::LogicError("Uncached read texture type");
    }
    return it->second;
}

Shader::TextureType FileEnvironment::ReadTextureType(u32 handle) {
    const auto it{texture_types.find(handle)};
    if (it == texture_types.end()) {
        throw Shader::LogicError("Uncached read texture type");
    }
    return it->second;
}

Shader::TexturePixelFormat FileEnvironment::ReadTexturePixelFormat(u32 handle) {
    const auto it{texture_pixel_formats.find(handle)};
    if (it == texture_pixel_formats.end()) {
        throw Shader::LogicError("Uncached read texture pixel format");
    }
    return it->second;
}

bool FileEnvironment::IsTexturePixelFormatInteger(u32 handle) {
    return VideoCore::Surface::IsPixelFormatInteger(
        static_cast<VideoCore::Surface::PixelFormat>(ReadTexturePixelFormat(handle)));
}

u32 FileEnvironment::ReadViewportTransformState() {
    return viewport_transform_state;
}

u32 FileEnvironment::LocalMemorySize() const {
    return local_memory_size;
}

u32 FileEnvironment::SharedMemorySize() const {
    return shared_memory_size;
}

u32 FileEnvironment::TextureBoundBuffer() const {
    return texture_bound;
}

std::array<u32, 3> FileEnvironment::WorkgroupSize() const {
    return workgroup_size;
}

std::optional<Shader::ReplaceConstant> FileEnvironment::GetReplaceConstBuffer(u32 bank,
                                                                              u32 offset) {
    const u64 key = (static_cast<u64>(bank) << 32) | static_cast<u64>(offset);
    auto it = cbuf_replacements.find(key);
    if (it == cbuf_replacements.end()) {
        return std::nullopt;
    }
    return it->second;
}

void SerializePipeline(std::span<const char> key, std::span<const GenericEnvironment* const> envs,
                       const std::filesystem::path& filename, u32 cache_version) try {
    std::ofstream file(filename, std::ios::binary | std::ios::ate | std::ios::app);
    file.exceptions(std::ifstream::failbit);
    if (!file.is_open()) {
        LOG_ERROR(Common_Filesystem, "Failed to open pipeline cache file {}",
                  Common::FS::PathToUTF8String(filename));
        return;
    }
    if (file.tellp() == 0) {
        // Write header
        file.write(MAGIC_NUMBER.data(), MAGIC_NUMBER.size())
            .write(reinterpret_cast<const char*>(&cache_version), sizeof(cache_version));
    }
    if (!std::ranges::all_of(envs, &GenericEnvironment::CanBeSerialized)) {
        return;
    }
    const u32 num_envs{static_cast<u32>(envs.size())};
    file.write(reinterpret_cast<const char*>(&num_envs), sizeof(num_envs));
    for (const GenericEnvironment* const env : envs) {
        env->Serialize(file);
    }
    file.write(key.data(), key.size_bytes());

} catch (const std::ios_base::failure& e) {
    LOG_ERROR(Common_Filesystem, "{}", e.what());
    if (!Common::FS::RemoveFile(filename)) {
        LOG_ERROR(Common_Filesystem, "Failed to delete pipeline cache file {}",
                  Common::FS::PathToUTF8String(filename));
    }
}

void LoadPipelines(
    std::stop_token stop_loading, const std::filesystem::path& filename, u32 expected_cache_version,
    Common::UniqueFunction<void, std::ifstream&, FileEnvironment> load_compute,
    Common::UniqueFunction<void, std::ifstream&, std::vector<FileEnvironment>> load_graphics) try {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return;
    }
    file.exceptions(std::ifstream::failbit);
    const auto end{file.tellg()};
    file.seekg(0, std::ios::beg);

    std::array<char, 8> magic_number;
    u32 cache_version;
    file.read(magic_number.data(), magic_number.size())
        .read(reinterpret_cast<char*>(&cache_version), sizeof(cache_version));
    if (magic_number != MAGIC_NUMBER || cache_version != expected_cache_version) {
        file.close();
        if (Common::FS::RemoveFile(filename)) {
            if (magic_number != MAGIC_NUMBER) {
                LOG_ERROR(Common_Filesystem, "Invalid pipeline cache file");
            }
            if (cache_version != expected_cache_version) {
                LOG_INFO(Common_Filesystem, "Deleting old pipeline cache");
            }
        } else {
            LOG_ERROR(Common_Filesystem,
                      "Invalid pipeline cache file and failed to delete it in \"{}\"",
                      Common::FS::PathToUTF8String(filename));
        }
        return;
    }
    while (file.tellg() != end) {
        if (stop_loading.stop_requested()) {
            return;
        }
        u32 num_envs{};
        file.read(reinterpret_cast<char*>(&num_envs), sizeof(num_envs));
        std::vector<FileEnvironment> envs(num_envs);
        for (FileEnvironment& env : envs) {
            env.Deserialize(file);
        }
        if (envs.front().ShaderStage() == Shader::Stage::Compute) {
            load_compute(file, std::move(envs.front()));
        } else {
            load_graphics(file, std::move(envs));
        }
    }

} catch (const std::ios_base::failure& e) {
    LOG_ERROR(Common_Filesystem, "{}", e.what());
    if (!Common::FS::RemoveFile(filename)) {
        LOG_ERROR(Common_Filesystem, "Failed to delete pipeline cache file {}",
                  Common::FS::PathToUTF8String(filename));
    }
}

} // namespace VideoCommon
