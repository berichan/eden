// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <bitset>
#include "common/assert.h"
#include "common/logging.h"
#include "core/core.h"
#include "shader_recompiler/exception.h"
#include "video_core/control/native_compute.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/memory_manager.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/textures/decoders.h"

namespace Tegra::Engines {

KeplerCompute::KeplerCompute(MemoryManager& memory_manager_)
    : memory_manager{memory_manager_}
    , upload_state{memory_manager, regs.upload}
{
    execution_mask.reset();
    execution_mask[KEPLER_COMPUTE_REG_INDEX(exec_upload)] = true;
    execution_mask[KEPLER_COMPUTE_REG_INDEX(data_upload)] = true;
    execution_mask[KEPLER_COMPUTE_REG_INDEX(launch)] = true;
}

KeplerCompute::~KeplerCompute() = default;

void KeplerCompute::SetNativeContext(
    std::shared_ptr<const Control::NativeGpuGraphicsContext> context) {
    ASSERT(!context || context->GetAddressSpace().get() == &memory_manager);
    method_sink.clear();
    uploads.clear();
    indirect_compute.reset();
    native_stop_token = {};
    native_compute =
        context ? std::make_unique<Control::NativeCompute>(std::move(context)) : nullptr;
    native_mode = false;
    execution_mask.reset();
    if (native_compute) {
        execution_mask.set();
    } else {
        execution_mask[KEPLER_COMPUTE_REG_INDEX(exec_upload)] = true;
        execution_mask[KEPLER_COMPUTE_REG_INDEX(data_upload)] = true;
        execution_mask[KEPLER_COMPUTE_REG_INDEX(launch)] = true;
    }
}

bool KeplerCompute::CallNativeMethod(u32 method, u32 value, bool last, std::stop_token stop_token) {
    if (!native_compute) {
        return false;
    }
    native_stop_token = stop_token;
    using Qmd = Control::NativeQmdScheduler;
    struct HostContext {
        KeplerCompute& owner;
        std::stop_token stop_token;
    } context{*this, stop_token};
    const Control::NativeCompute::Host host{
        {&context,
         [](void*, const Qmd::Release& release) {
             // Timestamped, wide-payload and reduction releases need their own adapter.
             return release.SizeBytes() == 4 && !(release.flags & ((1U << 26) | (1U << 29)));
         },
         [](void* context, const Qmd::Descriptor& descriptor) {
             return static_cast<HostContext*>(context)->owner.DispatchNativeCompute(descriptor);
         },
         [](void* context, const Qmd::Release& release) {
             auto& self = static_cast<HostContext*>(context)->owner;
             const auto token = static_cast<HostContext*>(context)->stop_token;
             if (token.stop_requested() || !self.rasterizer ||
                 !self.rasterizer->SynchronizeNativeWork() || token.stop_requested()) {
                 return false;
             }
             const u32 payload = static_cast<u32>(release.payload);
             return self.memory_manager.WriteBlockChecked(
                 release.address, {reinterpret_cast<const u8*>(&payload), sizeof(payload)});
         },
         [](void* context) {
             return static_cast<HostContext*>(context)->stop_token.stop_requested();
         }},
        [](void* context) {
            const auto& self = static_cast<HostContext*>(context)->owner;
            return self.rasterizer && self.rasterizer->SynchronizeNativeWork();
        },
        [](void* context, const Control::NativeCacheInvalidation& action) {
            const auto& self = static_cast<HostContext*>(context)->owner;
            return self.rasterizer && self.rasterizer->InvalidateNativeCaches(action);
        },
    };
    return native_compute->CallMethod(method, value, last, host);
}

bool KeplerCompute::HasNativeMethodFailure() const {
    return native_compute && native_compute->GetError() != Control::NativeCompute::Error::None;
}

bool KeplerCompute::HasPendingNativeMacro() const {
    return native_compute && native_compute->HasPendingMacro();
}

bool KeplerCompute::HasPendingNativeInlineUpload() const {
    return native_compute && native_compute->HasPendingInlineUpload();
}

void KeplerCompute::BindRasterizer(VideoCore::RasterizerInterface* rasterizer_) {
    rasterizer = rasterizer_;
    upload_state.BindRasterizer(rasterizer);
}

void KeplerCompute::ConsumeSinkImpl(Core::System& system) {
    for (auto [method, value] : method_sink) {
        regs.reg_array[method] = value;
    }
    method_sink.clear();
}

void KeplerCompute::CallMethod(Core::System& system, u32 method, u32 method_argument, bool is_last_call) {
    if (native_compute) {
        CallNativeMethod(method, method_argument, is_last_call);
        return;
    }
    ASSERT_MSG(method < Regs::NUM_REGS, "Invalid KeplerCompute register, increase the size of the Regs structure");

    regs.reg_array[method] = method_argument;

    switch (method) {
    case KEPLER_COMPUTE_REG_INDEX(exec_upload): {
        UploadInfo info{.upload_address = upload_address,
                        .exec_address = upload_state.ExecTargetAddress(),
                        .copy_size = upload_state.GetUploadSize(),
                        .was_dirty = upload_dirty};
        uploads.push_back(info);
        upload_state.ProcessExec(regs.exec_upload.linear != 0);
        break;
    }
    case KEPLER_COMPUTE_REG_INDEX(data_upload): {
        upload_address = current_dma_segment;
        upload_dirty = current_dirty;
        current_dirty = false;
        upload_state.ProcessData(method_argument, is_last_call);
        break;
    }
    case KEPLER_COMPUTE_REG_INDEX(launch): {
        const GPUVAddr launch_desc_loc = regs.launch_desc_loc.Address();

        for (auto& data : uploads) {
            const GPUVAddr offset = data.exec_address - launch_desc_loc;
            if (offset / sizeof(u32) == LAUNCH_REG_INDEX(grid_dim_x)) {
                const bool source_dirty = memory_manager.IsMemoryDirty(data.upload_address, data.copy_size);
                if (data.was_dirty || source_dirty) {
                    indirect_compute = {data.upload_address};
                }
            }
        }
        uploads.clear();
        ProcessLaunch();
        indirect_compute = std::nullopt;
        break;
    }
    default:
        break;
    }
}

void KeplerCompute::CallMultiMethod(Core::System& system, u32 method, const u32* base_start, u32 amount, u32 methods_pending) {
    if (native_compute) {
        for (u32 index = 0; index < amount; ++index) {
            if (!CallNativeMethod(method, base_start[index], methods_pending - index <= 1)) {
                break;
            }
        }
        return;
    }
    switch (method) {
    case KEPLER_COMPUTE_REG_INDEX(data_upload):
        upload_address = current_dma_segment;
        upload_dirty = current_dirty;
        current_dirty = false;
        upload_state.ProcessData(base_start, amount);
        return;
    default:
        for (u32 i = 0; i < amount; i++) {
            CallMethod(system, method, base_start[i], methods_pending - i <= 1);
        }
        break;
    }
}

void KeplerCompute::ProcessLaunch() {
    native_mode = false;
    const GPUVAddr launch_desc_loc = regs.launch_desc_loc.Address();
    memory_manager.ReadBlockUnsafe(launch_desc_loc, &launch_description,
                                   LaunchParams::NUM_LAUNCH_PARAMETERS * sizeof(u32));
    rasterizer->DispatchCompute();
}

bool KeplerCompute::ConfigureNativeLaunch(std::span<const u32, 64> descriptor) {
    const auto& qmd = descriptor;
    const GPUVAddr program = (GPUVAddr{qmd[49] & 0x1ffff} << 32) | qmd[48];
    const u32 block_x = qmd[18] >> 16;
    const u32 block_y = qmd[19] & 0xffff;
    const u32 block_z = qmd[19] >> 16;
    const bool nonempty = qmd[12] && (qmd[13] & 0xffff) && (qmd[14] & 0xffff);
    if ((qmd[18] & 0xff) != 0x30 || (qmd[4] & 0x100) || (qmd[12] & 0x80000000) ||
        !memory_manager.IsWithinGPUAddressRange(program) || (qmd[23] & 0xffffff) > 0xfffff ||
        (qmd[50] & 0xffffff) > 0xfffff ||
        (nonempty &&
         (!block_x || !block_y || !block_z || u64{block_x} * block_y * block_z > 1024))) {
        return false;
    }
    LaunchParams launch{};
    launch.grid_dim_x.Assign(qmd[12]);
    launch.grid_dim_y.Assign(qmd[13] & 0xffff);
    launch.grid_dim_z.Assign(qmd[14] & 0xffff);
    launch.shared_alloc.Assign(qmd[17] & 0x3ffff);
    launch.block_dim_x.Assign(block_x);
    launch.block_dim_y.Assign(block_y);
    launch.block_dim_z.Assign(block_z);
    launch.const_buffer_enable_mask.Assign(qmd[20] & 0xff);
    launch.local_pos_alloc.Assign(qmd[23] & 0xffffff);
    launch.local_crs_alloc.Assign(qmd[50] & 0xffffff);
    for (u32 slot = 0; slot < NumConstBuffers; ++slot) {
        if (!(launch.const_buffer_enable_mask.Value() & (1U << slot))) {
            continue;
        }
        const u32 high = qmd[33 + slot * 2];
        const GPUVAddr address = (GPUVAddr{high & 0x1ffff} << 32) | qmd[32 + slot * 2];
        const u32 size = ((high >> 19) & 0x1fff) << 4;
        if ((high & 0x1ff00) || !memory_manager.IsWithinGPUAddressRange(address) ||
            (size != 0 && !memory_manager.IsWithinGPUAddressRange(address + size - 1))) {
            return false;
        }
        auto& buffer = launch.const_buffer_config[slot];
        buffer.address_low = qmd[32 + slot * 2];
        buffer.address_high.Assign(high & 0xff);
        buffer.size.Assign(size);
    }
    launch_description = launch;
    regs.code_loc.address_low = qmd[48];
    regs.code_loc.address_high = qmd[49] & 0x1ffff;
    indirect_compute.reset();
    native_mode = true;
    return true;
}

bool KeplerCompute::DispatchNativeCompute(std::span<const u32, 64> descriptor) {
    if (!ConfigureNativeLaunch(descriptor)) {
        return false;
    }
    if (!launch_description.grid_dim_x || !launch_description.grid_dim_y ||
        !launch_description.grid_dim_z) {
        return true;
    }
    if (!rasterizer) {
        return false;
    }
    for (u32 slot = 0; slot < NumConstBuffers; ++slot) {
        if ((launch_description.const_buffer_enable_mask.Value() & (1U << slot)) &&
            !memory_manager.IsReadableRange(launch_description.const_buffer_config[slot].Address(),
                                            launch_description.const_buffer_config[slot].size)) {
            return false;
        }
    }
    try {
        return rasterizer->DispatchNativeCompute();
    } catch (const Shader::Exception& error) {
        LOG_ERROR(HW_GPU, "Native compute shader rejected: {}", error.what());
        return false;
    }
}

bool KeplerCompute::ReadNativeConstantBuffer(u32 index, u32 offset, std::span<u8> output) const {
    if (!native_mode || index >= NumConstBuffers ||
        !(launch_description.const_buffer_enable_mask.Value() & (1U << index))) {
        return false;
    }
    const auto& buffer = launch_description.const_buffer_config[index];
    if (offset > buffer.size || output.size() > buffer.size.Value() - offset) {
        return false;
    }
    return memory_manager.ReadBlockChecked(buffer.Address() + offset, output);
}

bool KeplerCompute::IsNativeLaunchSupported(std::array<u32, 3> max_grid,
                                            std::array<u32, 3> max_block, u32 max_invocations,
                                            u32 max_shared_memory) const {
    if (!native_mode || launch_description.shared_alloc > max_shared_memory) {
        return false;
    }
    const auto& launch = launch_description;
    const std::array<u32, 3> grid{launch.grid_dim_x, launch.grid_dim_y, launch.grid_dim_z};
    const std::array<u32, 3> block{launch.block_dim_x, launch.block_dim_y, launch.block_dim_z};
    for (u32 axis = 0; axis < 3; ++axis) {
        if (grid[axis] > max_grid[axis] || block[axis] > max_block[axis]) {
            return false;
        }
    }
    return u64{block[0]} * block[1] * block[2] <= max_invocations;
}

Texture::TICEntry KeplerCompute::GetTICEntry(u32 tic_index) const {
    const GPUVAddr tic_address_gpu{regs.tic.Address() + tic_index * sizeof(Texture::TICEntry)};

    Texture::TICEntry tic_entry;
    memory_manager.ReadBlockUnsafe(tic_address_gpu, &tic_entry, sizeof(Texture::TICEntry));
    return tic_entry;
}

Texture::TSCEntry KeplerCompute::GetTSCEntry(u32 tsc_index) const {
    const GPUVAddr tsc_address_gpu{regs.tsc.Address() + tsc_index * sizeof(Texture::TSCEntry)};

    Texture::TSCEntry tsc_entry;
    memory_manager.ReadBlockUnsafe(tsc_address_gpu, &tsc_entry, sizeof(Texture::TSCEntry));
    return tsc_entry;
}

} // namespace Tegra::Engines
