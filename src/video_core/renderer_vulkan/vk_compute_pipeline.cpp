// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstring>
#include <vector>

#include <boost/container/small_vector.hpp>
#include <fmt/format.h>

#include "video_core/renderer_vulkan/pipeline_helper.h"
#include "video_core/renderer_vulkan/pipeline_statistics.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_descriptor_pool.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/shader_notify.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"
#include "video_core/gpu_logging/gpu_logging.h"
#include "common/logging.h"
#include "common/settings.h"

namespace Vulkan {

using Shader::ImageBufferDescriptor;
using Shader::Backend::SPIRV::RESCALING_LAYOUT_WORDS_OFFSET;
using Tegra::Texture::TexturePair;

ComputePipeline::ComputePipeline(const Device& device_, Scheduler& scheduler, vk::PipelineCache& pipeline_cache_,
                                 DescriptorPool& descriptor_pool,
                                 GuestDescriptorQueue& guest_descriptor_queue_,
                                 DescriptorBufferRing& descriptor_buffer_ring_,
                                 Common::ThreadWorker* thread_worker,
                                 PipelineStatistics* pipeline_statistics,
                                 VideoCore::ShaderNotify* shader_notify, const Shader::Info& info_,
                                 vk::ShaderModule spv_module_, u64 shader_hash_)
    : device{device_},
      pipeline_cache(pipeline_cache_), guest_descriptor_queue{guest_descriptor_queue_},
      descriptor_buffer_ring{descriptor_buffer_ring_}, info{info_},
      shader_hash{shader_hash_}, spv_module(std::move(spv_module_)) {
    if (shader_notify) {
        shader_notify->MarkShaderBuilding();
    }
    std::copy_n(info.constant_buffer_used_sizes.begin(), uniform_buffer_sizes.size(),
                uniform_buffer_sizes.begin());
    num_descriptor_entries = NumDescriptorEntries(info);

    DescriptorLayoutBuilder builder{device};
    builder.Add(info, VK_SHADER_STAGE_COMPUTE_BIT);

    uses_push_descriptor = builder.CanUsePushDescriptor();
    uses_descriptor_buffer = builder.CanUseDescriptorBuffer() && descriptor_buffer_ring.IsValid();
    descriptor_set_layout =
        builder.CreateDescriptorSetLayout(uses_push_descriptor, uses_descriptor_buffer);
    if (uses_descriptor_buffer) {
        descriptor_buffer_layout = builder.MakeDescriptorBufferLayout(*descriptor_set_layout);
        if (!descriptor_buffer_ring.CanAllocate(descriptor_buffer_layout.size)) {
            LOG_DEBUG(Render_Vulkan,
                        "Compute shader {:016X} needs {} descriptor bytes per dispatch, falling "
                        "back to sets",
                        shader_hash, descriptor_buffer_layout.size);
            uses_descriptor_buffer = false;
            descriptor_buffer_layout = {};
            descriptor_set_layout = builder.CreateDescriptorSetLayout(false);
        }
    }
    pipeline_layout = builder.CreatePipelineLayout(*descriptor_set_layout);
    if (!uses_descriptor_buffer) {
        descriptor_update_template =
            builder.CreateTemplate(*descriptor_set_layout, *pipeline_layout, uses_push_descriptor);
        if (!uses_push_descriptor) {
            descriptor_allocator =
                descriptor_pool.Allocator(device, scheduler, *descriptor_set_layout, info);
        }
    }

    auto func{[this, shader_notify, pipeline_statistics] {
        const VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT subgroup_size_ci{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT,
            .pNext = nullptr,
            .requiredSubgroupSize = GuestWarpSize,
        };
        VkPipelineCreateFlags flags{};
        if (device.IsKhrPipelineExecutablePropertiesEnabled() && Settings::values.renderer_debug.GetValue()) {
            flags |= VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
        }
        if (uses_descriptor_buffer) {
            flags |= VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
        }
        const VkComputePipelineCreateInfo compute_ci{
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .pNext = nullptr,
            .flags = flags,
            .stage{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .pNext =
                    device.IsExtSubgroupSizeControlSupported() ? &subgroup_size_ci : nullptr,
                .flags = 0,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = *spv_module,
                .pName = "main",
                .pSpecializationInfo = nullptr,
            },
            .layout = *pipeline_layout,
            .basePipelineHandle = 0,
            .basePipelineIndex = 0,
        };
        try {
            pipeline = device.GetLogical().CreateComputePipeline(compute_ci, *pipeline_cache);
        } catch (const vk::Exception& exception) {
            LOG_CRITICAL(Render_Vulkan, "Adreno rejected compute shader {:016X}: {}", shader_hash,
                         exception.what());
            std::scoped_lock lock{build_mutex};
            is_built = true;
            build_condvar.notify_one();
            if (shader_notify) {
                shader_notify->MarkShaderComplete();
            }
            return;
        }

        // Log compute pipeline creation
        if (GPU::Logging::IsActive()) {
            GPU::Logging::GPULogger::GetInstance().LogPipelineStateChange(
                "ComputePipeline created"
            );
        }

        if (pipeline_statistics) {
            pipeline_statistics->Collect(device, *pipeline);
        }
        std::scoped_lock lock{build_mutex};
        is_built = true;
        build_condvar.notify_one();
        if (shader_notify) {
            shader_notify->MarkShaderComplete();
        }
    }};
    if (thread_worker) {
        thread_worker->QueueWork(std::move(func));
    } else {
        func();
    }
}

std::optional<ComputePipeline::NativeBindings> ComputePipeline::SnapshotNativeBindings(
    Tegra::Engines::KeplerCompute& kepler_compute) const {
    if (!kepler_compute.IsNativeMode() || (info.constant_buffer_mask & ~0xffU) ||
        !info.texture_buffer_descriptors.empty() || !info.image_buffer_descriptors.empty() ||
        !info.texture_descriptors.empty() || !info.image_descriptors.empty() ||
        info.storage_buffers_descriptors.size() > VideoCommon::NUM_STORAGE_BUFFERS) {
        return std::nullopt;
    }
    NativeBindings snapshot;
    const auto& qmd = kepler_compute.launch_description;
    std::vector<u8> visible_uniform;
    for (size_t index = 0; index < snapshot.uniforms.size(); ++index) {
        if (!(info.constant_buffer_mask & (1U << index))) {
            continue;
        }
        if (!(qmd.const_buffer_enable_mask.Value() & (1U << index))) {
            return std::nullopt;
        }
        const auto& cbuf = qmd.const_buffer_config[index];
        auto& uniform = snapshot.uniforms[index];
        uniform.address = cbuf.Address();
        uniform.size = (std::min)(cbuf.size.Value(), uniform_buffer_sizes[index]);
        if (uniform.size == 0 || uniform.size > 0x10000 ||
            kepler_compute.GetNativeStopToken().stop_requested()) {
            return std::nullopt;
        }
        // The unaligned uniform-buffer streaming path copies CPU backing directly.
        // Download GPU-written bytes now, before either mapping or cache locks are held.
        visible_uniform.resize(uniform.size);
        if (!kepler_compute.ReadNativeConstantBuffer(static_cast<u32>(index), 0, visible_uniform)) {
            return std::nullopt;
        }
    }
    for (const auto& desc : info.storage_buffers_descriptors) {
        if (desc.count != 1) {
            return std::nullopt;
        }
        std::array<u8, 16> bytes{};
        const bool has_tail =
            kepler_compute.ReadNativeConstantBuffer(desc.cbuf_index, desc.cbuf_offset, bytes);
        if (!has_tail && !kepler_compute.ReadNativeConstantBuffer(desc.cbuf_index, desc.cbuf_offset,
                                                                  std::span{bytes}.first(8))) {
            return std::nullopt;
        }
        NativeBindings::Storage storage{};
        std::memcpy(&storage.address, bytes.data(), sizeof(storage.address));
        if (has_tail) {
            u64 hint{};
            std::memcpy(&hint, bytes.data() + 8, sizeof(hint));
            storage.size_hint = hint;
        }
        storage.written = desc.is_written;
        snapshot.storage.push_back(storage);
    }
    return snapshot;
}

bool ComputePipeline::ResolveNativeBindings(NativeBindings& bindings,
                                            const Tegra::MemoryManager::MappingLease& lease,
                                            std::stop_token stop_token) const {
    bindings.resolved = false;
    for (auto& uniform : bindings.uniforms) {
        if (stop_token.stop_requested()) {
            return false;
        }
        if (uniform.size == 0) {
            continue;
        }
        const auto device_address = lease.ResolveContiguous(uniform.address, uniform.size, false);
        if (!device_address || *device_address == 0) {
            return false;
        }
        uniform.device_address = *device_address;
    }
    const u64 alignment = device.GetStorageBufferAlignment();
    const u64 max_range = (std::min)(device.GetMaxStorageBufferRange(), u64{0xffffffff});
    if (alignment == 0 || (alignment & (alignment - 1))) {
        return false;
    }
    for (auto& storage : bindings.storage) {
        if (stop_token.stop_requested()) {
            return false;
        }
        storage.binding = {};
        if (storage.address == 0) {
            continue;
        }
        if (storage.address >= (u64{1} << 40)) {
            return false;
        }
        const GPUVAddr aligned_address = storage.address & ~(alignment - 1);
        const u64 prefix = storage.address - aligned_address;
        if (prefix >= max_range) {
            return false;
        }
        // Preserve the bounded legacy size hint for generic CBUF pointers. A
        // neighboring qword is not universally a size-bearing descriptor ABI.
        const u32 hint = storage.size_hint && (*storage.size_hint >> 32) == 0
                             ? static_cast<u32>(*storage.size_hint)
                             : 0;
        const u64 limit = (std::min)(max_range - prefix, (std::max)(u64{8U << 20}, u64{hint}));
        const u64 mapped =
            lease.ContiguousPrefix(storage.address, limit, storage.written, stop_token);
        const u64 selected_size =
            hint != 0 && hint <= mapped ? hint : (std::min)(mapped, u64{8U << 20});
        if (selected_size == 0) {
            return false;
        }
        const u64 binding_size = prefix + selected_size;
        const auto device_address =
            lease.ResolveContiguous(aligned_address, binding_size, storage.written);
        if (!device_address || *device_address == 0 || (*device_address % alignment) != 0) {
            return false;
        }
        // Zero gpu_addr prevents the cache from re-resolving this leased physical range.
        storage.binding = {.device_addr = *device_address, .size = static_cast<u32>(binding_size)};
    }
    bindings.resolved = true;
    return true;
}

bool ComputePipeline::Configure(Tegra::Engines::KeplerCompute& kepler_compute,
                                Tegra::MemoryManager& gpu_memory, Scheduler& scheduler,
                                BufferCache& buffer_cache, TextureCache& texture_cache,
                                const NativeBindings* native_bindings) {
    if (kepler_compute.IsNativeMode() &&
        (!native_bindings || !native_bindings->resolved ||
         native_bindings->storage.size() != info.storage_buffers_descriptors.size())) {
        return false;
    }
    if (native_bindings && !kepler_compute.IsNativeMode()) {
        return false;
    }
    guest_descriptor_queue.Acquire(scheduler, num_descriptor_entries, uses_descriptor_buffer);

    buffer_cache.SetComputeUniformBufferState(info.constant_buffer_mask, &uniform_buffer_sizes);
    buffer_cache.UnbindComputeStorageBuffers();
    if (native_bindings) {
        std::array<VideoCommon::Binding, 8> uniforms{};
        for (size_t index = 0; index < uniforms.size(); ++index) {
            const auto& uniform = native_bindings->uniforms[index];
            uniforms[index] = {.device_addr = uniform.device_address, .size = uniform.size};
        }
        buffer_cache.SetNativeComputeUniformBuffers(uniforms);
        for (size_t index = 0; index < native_bindings->storage.size(); ++index) {
            const auto& storage = native_bindings->storage[index];
            if (!buffer_cache.BindNativeComputeStorageBuffer(index, storage.binding,
                                                             storage.written)) {
                return false;
            }
        }
    } else {
        size_t ssbo_index{};
        for (const auto& desc : info.storage_buffers_descriptors) {
            ASSERT(desc.count == 1);
            buffer_cache.BindComputeStorageBuffer(ssbo_index, desc.cbuf_index, desc.cbuf_offset,
                                                  desc.is_written);
            ++ssbo_index;
        }
    }

    texture_cache.SynchronizeDescriptors(true);

    boost::container::small_vector<VideoCommon::ImageViewInOut, 64> views;
    boost::container::small_vector<VideoCommon::SamplerId, 64> samplers;

    const auto& qmd{kepler_compute.launch_description};
    const auto& cbufs{qmd.const_buffer_config};
    const bool via_header_index{qmd.linked_tsc != 0};
    const auto read_handle{[&](const auto& desc, u32 index) {
        ASSERT(((qmd.const_buffer_enable_mask >> desc.cbuf_index) & 1) != 0);
        const u32 index_offset{index << desc.size_shift};
        const u32 offset{desc.cbuf_offset + index_offset};
        const GPUVAddr addr{cbufs[desc.cbuf_index].Address() + offset};
        if constexpr (std::is_same_v<decltype(desc), const Shader::TextureDescriptor&> ||
                      std::is_same_v<decltype(desc), const Shader::TextureBufferDescriptor&>) {
            if (desc.has_secondary) {
                ASSERT(((qmd.const_buffer_enable_mask >> desc.secondary_cbuf_index) & 1) != 0);
                const u32 secondary_offset{desc.secondary_cbuf_offset + index_offset};
                const GPUVAddr separate_addr{cbufs[desc.secondary_cbuf_index].Address() +
                                             secondary_offset};
                const u32 lhs_raw{gpu_memory.Read<u32>(addr) << desc.shift_left};
                const u32 rhs_raw{gpu_memory.Read<u32>(separate_addr) << desc.secondary_shift_left};
                return TexturePair(lhs_raw | rhs_raw, via_header_index);
            }
        }
        return TexturePair(gpu_memory.Read<u32>(addr), via_header_index);
    }};
    const auto add_image{[&](const auto& desc, bool blacklist) {
        for (u32 index = 0; index < desc.count; ++index) {
            const auto handle{read_handle(desc, index)};
            views.push_back({
                .index = handle.first,
                .blacklist = blacklist,
                .id = {},
            });
        }
    }};
    for (const auto& desc : info.texture_buffer_descriptors) {
        add_image(desc, false);
    }
    for (const auto& desc : info.image_buffer_descriptors) {
        add_image(desc, false);
    }
    for (const auto& desc : info.texture_descriptors) {
        for (u32 index = 0; index < desc.count; ++index) {
            const auto handle{read_handle(desc, index)};
            views.push_back({handle.first});

            VideoCommon::SamplerId sampler = texture_cache.GetSamplerId(handle.second, true);
            samplers.push_back(sampler);
        }
    }
    for (const auto& desc : info.image_descriptors) {
        add_image(desc, desc.is_written);
    }
    texture_cache.FillImageViews(std::span(views.data(), views.size()), true);

    buffer_cache.UnbindComputeTextureBuffers();
    size_t index{};
    const auto add_buffer{[&](const auto& desc) {
        constexpr bool is_image = std::is_same_v<decltype(desc), const ImageBufferDescriptor&>;
        for (u32 i = 0; i < desc.count; ++i) {
            bool is_written{false};
            if constexpr (is_image) {
                is_written = desc.is_written;
            }
            ImageView& image_view = texture_cache.GetImageView(views[index].id);
            PixelFormat format{image_view.format};
            if constexpr (is_image) {
                if (const auto explicit_format{PixelFormatFromImageFormat(desc.format)}) {
                    format = *explicit_format;
                }
            }
            buffer_cache.BindComputeTextureBuffer(index, image_view.GpuAddr(),
                                                  image_view.BufferSize(), format,
                                                  is_written, is_image);
            ++index;
        }
    }};
    std::ranges::for_each(info.texture_buffer_descriptors, add_buffer);
    std::ranges::for_each(info.image_buffer_descriptors, add_buffer);

    buffer_cache.UpdateComputeBuffers();
    buffer_cache.BindHostComputeBuffers();
    if (buffer_cache.any_buffer_uploaded) {
        buffer_cache.runtime.PostCopyBarrier();
        buffer_cache.any_buffer_uploaded = false;
    }

    RescalingPushConstant rescaling;
    const VideoCommon::SamplerId* samplers_it{samplers.data()};
    const VideoCommon::ImageViewInOut* views_it{views.data()};
    PushImageDescriptors(texture_cache, guest_descriptor_queue, info, rescaling, samplers_it,
                         views_it);

    if (!is_built.load(std::memory_order::relaxed)) {
        // Wait for the pipeline to be built
        scheduler.Record([this](vk::CommandBuffer) {
            std::unique_lock lock{build_mutex};
            build_condvar.wait(lock, [this] { return is_built.load(std::memory_order::relaxed); });
        });
    }

    // Log compute pipeline binding
    if (GPU::Logging::IsActive() &&
        Settings::values.gpu_log_vulkan_calls.GetValue()) {
        GPU::Logging::GPULogger::GetInstance().LogPipelineBind(true, "compute pipeline");
    }

    const DescriptorUpdateEntry* const descriptor_data{guest_descriptor_queue.UpdateData()};
    VkDeviceSize descriptor_buffer_offset{};
    u32 descriptor_buffer_chunk{};
    if (uses_descriptor_buffer) {
        const DescriptorBufferRing::Allocation alloc{
            descriptor_buffer_ring.Allocate(scheduler, descriptor_buffer_layout.size)};
        if (!alloc.host) {
            LOG_DEBUG(Render_Vulkan, "Failed to reserve descriptor memory, skipping dispatch");
            return false;
        }
        WriteDescriptorBuffer(device, descriptor_buffer_layout, descriptor_data, alloc.host);
        descriptor_buffer_offset = alloc.offset;
        descriptor_buffer_chunk = alloc.chunk;
    }

    const bool bind_descriptor_buffer{
        uses_descriptor_buffer && scheduler.UpdateDescriptorBufferChunk(descriptor_buffer_chunk)};

    const bool is_rescaling = !info.texture_descriptors.empty() || !info.image_descriptors.empty();
    scheduler.Record([this, descriptor_data, is_rescaling, descriptor_buffer_offset,
                      descriptor_buffer_chunk, bind_descriptor_buffer,
                      rescaling_data = rescaling.Data()](vk::CommandBuffer cmdbuf) {
        if (bind_descriptor_buffer) {
            const VkDescriptorBufferBindingInfoEXT binding_info{
                descriptor_buffer_ring.BindingInfo(descriptor_buffer_chunk)};
            cmdbuf.BindDescriptorBuffersEXT(binding_info);
        }
        if (!pipeline) {
            return;
        }
        cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline);
        if (!descriptor_set_layout) {
            return;
        }
        if (is_rescaling) {
            cmdbuf.PushConstants(*pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                                 RESCALING_LAYOUT_WORDS_OFFSET, sizeof(rescaling_data),
                                 rescaling_data.data());
        }
        if (uses_descriptor_buffer) {
            const u32 buffer_index{};
            cmdbuf.SetDescriptorBufferOffsetsEXT(VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline_layout,
                                                 0, buffer_index, descriptor_buffer_offset);
        } else if (uses_push_descriptor) {
            cmdbuf.PushDescriptorSetWithTemplateKHR(*descriptor_update_template, *pipeline_layout,
                                                    0, descriptor_data);
        } else {
            const VkDescriptorSet descriptor_set{descriptor_allocator.Commit()};
            const vk::Device& dev{device.GetLogical()};
            dev.UpdateDescriptorSet(descriptor_set, *descriptor_update_template, descriptor_data);
            cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline_layout, 0,
                                      descriptor_set, nullptr);
        }
    });
    return true;
}

} // namespace Vulkan
