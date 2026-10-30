// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>

#include <fmt/format.h>

#include "video_core/renderer_vulkan/renderer_vulkan.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/cityhash.h"
#include "common/logging.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/control/channel_state.h"
#include "video_core/control/native_cache.h"
#include "video_core/control/native_color_clear.h"
#include "video_core/control/native_indexed_draw.h"
#include "video_core/control/native_render_targets.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/gpu_logging/gpu_logging.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/renderer_vulkan/blit_image.h"
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"
#include "video_core/renderer_vulkan/maxwell_to_vk.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_descriptor_pool.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_query_cache.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_staging_buffer_pool.h"
#include "video_core/renderer_vulkan/vk_state_tracker.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/shader_cache.h"
#include "video_core/shader_environment.h"
#include "video_core/texture_cache/decode_bc.h"
#include "video_core/texture_cache/texture_cache_base.h"
#include "video_core/textures/astc.h"
#include "video_core/textures/decoders.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

using Maxwell = Tegra::Engines::Maxwell3D::Regs;
using VideoCommon::ImageViewId;
using VideoCommon::ImageViewType;


namespace {
struct DrawParams {
    u32 base_instance;
    u32 num_instances;
    u32 base_vertex;
    u32 num_vertices;
    u32 first_index;
    bool is_indexed;
};

VkViewport GetViewportState(const Device& device, const Maxwell& regs, size_t index, float scale) {
    const auto& src = regs.viewport_transform[index];
    const auto conv = [scale](float value) {
        float new_value = value * scale;
        if (scale < 1.0f) {
            const bool sign = std::signbit(value);
            new_value = std::round(std::abs(new_value));
            new_value = sign ? -new_value : new_value;
        }
        return new_value;
    };
    const float x = conv(src.translate_x - src.scale_x);
    const float width = conv(src.scale_x * 2.0f);
    float y = conv(src.translate_y - src.scale_y);
    float height = conv(src.scale_y * 2.0f);

    const bool lower_left = regs.window_origin.mode != Maxwell::WindowOrigin::Mode::UpperLeft;
    const bool y_negate = !device.IsNvViewportSwizzleSupported() &&
                          src.swizzle.y == Maxwell::ViewportSwizzle::NegativeY;

    if (lower_left) {
        // Flip by surface clip height
        y += conv(static_cast<f32>(regs.surface_clip.height));
        height = -height;
    }

    if (y_negate) {
        // Flip by viewport height
        y += height;
        height = -height;
    }

    const float reduce_z = regs.depth_mode == Maxwell::DepthMode::MinusOneToOne ? 1.0f : 0.0f;
    VkViewport viewport{
        .x = x,
        .y = y,
        .width = width != 0.0f ? width : 1.0f,
        .height = height != 0.0f ? height : 1.0f,
        .minDepth = src.translate_z - src.scale_z * reduce_z,
        .maxDepth = src.translate_z + src.scale_z,
    };
    if (!device.IsExtDepthRangeUnrestrictedSupported()) {
        viewport.minDepth = std::clamp(viewport.minDepth, 0.0f, 1.0f);
        viewport.maxDepth = std::clamp(viewport.maxDepth, 0.0f, 1.0f);
    }
    return viewport;
}

VkRect2D GetScissorState(const Maxwell& regs, size_t index, u32 up_scale = 1, u32 down_shift = 0) {
    const auto& src = regs.scissor_test[index];
    VkRect2D scissor{};
    const auto scale_up = [&](s32 value) -> s32 {
        if (value == 0) {
            return 0U;
        }
        const s32 upset = value * up_scale;
        s32 acumm = 0;
        if ((up_scale >> down_shift) == 0) {
            acumm = upset % 2;
        }
        const s32 converted_value = (value * up_scale) >> down_shift;
        return value < 0 ? std::min<s32>(converted_value - acumm, -1)
                         : std::max<s32>(converted_value + acumm, 1);
    };

    const bool lower_left = regs.window_origin.mode != Maxwell::WindowOrigin::Mode::UpperLeft;
    const s32 clip_height = regs.surface_clip.height;

    // Flip coordinates if lower left
    s32 min_y = lower_left ? (clip_height - src.max_y) : src.min_y.Value();
    s32 max_y = lower_left ? (clip_height - src.min_y) : src.max_y.Value();

    // Bound to render area
    min_y = (std::max)(min_y, 0);
    max_y = (std::max)(max_y, 0);

    if (src.enable) {
        scissor.offset.x = scale_up(src.min_x);
        scissor.offset.y = scale_up(min_y);
        scissor.extent.width = scale_up(src.max_x - src.min_x);
        scissor.extent.height = scale_up(max_y - min_y);
    } else {
        scissor.offset.x = 0;
        scissor.offset.y = 0;
        scissor.extent.width = (std::numeric_limits<s32>::max)();
        scissor.extent.height = (std::numeric_limits<s32>::max)();
    }
    return scissor;
}

DrawParams MakeDrawParams(const Tegra::Engines::Maxwell3D::DrawManager::State& draw_state, u32 num_instances, bool is_indexed) {
    DrawParams params{
        .base_instance = draw_state.base_instance,
        .num_instances = num_instances,
        .base_vertex = is_indexed ? draw_state.base_index : draw_state.vertex_buffer.first,
        .num_vertices = is_indexed ? draw_state.index_buffer.count : draw_state.vertex_buffer.count,
        .first_index = is_indexed ? draw_state.index_buffer.first : 0,
        .is_indexed = is_indexed,
    };
    // 6 triangle vertices per quad, base vertex is part of the index
    // See BindQuadIndexBuffer for more details
    if (draw_state.topology == Maxwell::PrimitiveTopology::Quads) {
        params.num_vertices = (params.num_vertices / 4) * 6;
        params.base_vertex = 0;
        params.is_indexed = true;
    } else if (draw_state.topology == Maxwell::PrimitiveTopology::QuadStrip) {
        params.num_vertices = (params.num_vertices - 2) / 2 * 6;
        params.base_vertex = 0;
        params.is_indexed = true;
    }
    return params;
}

bool SupportsPrimitiveRestart(VkPrimitiveTopology topology) {
    switch (topology) {
    case VK_PRIMITIVE_TOPOLOGY_POINT_LIST:
    case VK_PRIMITIVE_TOPOLOGY_LINE_LIST:
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST:
    case VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY:
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY:
    case VK_PRIMITIVE_TOPOLOGY_PATCH_LIST:
        return false;
    default:
        return true;
    }
}

bool IsPrimitiveRestartSupported(const Device& device, VkPrimitiveTopology topology) {
    return ((topology != VK_PRIMITIVE_TOPOLOGY_PATCH_LIST &&
             device.IsTopologyListPrimitiveRestartSupported()) ||
            SupportsPrimitiveRestart(topology) ||
            (topology == VK_PRIMITIVE_TOPOLOGY_PATCH_LIST &&
             device.IsPatchListPrimitiveRestartSupported()));
}
} // Anonymous namespace

RasterizerVulkan::RasterizerVulkan(Core::Frontend::EmuWindow& emu_window_, Tegra::GPU& gpu_,
                                   Tegra::MaxwellDeviceMemoryManager& device_memory_,
                                   const Device& device_, MemoryAllocator& memory_allocator_,
                                   StateTracker& state_tracker_, Scheduler& scheduler_)
    : gpu{gpu_}, device_memory{device_memory_}, device{device_},
      memory_allocator{memory_allocator_}, state_tracker{state_tracker_}, scheduler{scheduler_},
      staging_pool(device, memory_allocator, scheduler), descriptor_pool(device, scheduler),
      guest_descriptor_queue(device, UpdateDescriptorQueue::GUEST_FRAME_PAYLOAD_SIZE,
                             device.IsExtDescriptorBufferSupported()),
      compute_pass_descriptor_queue(device, UpdateDescriptorQueue::COMPUTE_FRAME_PAYLOAD_SIZE),
      descriptor_buffer_ring(device, memory_allocator),
      blit_image(device, scheduler, state_tracker, descriptor_pool), render_pass_cache(device),
      texture_cache_runtime{
          device,     scheduler,         memory_allocator, staging_pool,
          blit_image, render_pass_cache, descriptor_pool,  compute_pass_descriptor_queue},
      texture_cache(texture_cache_runtime, device_memory),
      buffer_cache_runtime(device, memory_allocator, scheduler, staging_pool,
                           guest_descriptor_queue, compute_pass_descriptor_queue, descriptor_pool),
      buffer_cache(device_memory, buffer_cache_runtime),
      query_cache_runtime(this, device_memory, buffer_cache, device, memory_allocator, scheduler,
                          staging_pool, compute_pass_descriptor_queue, descriptor_pool, texture_cache),
      query_cache(gpu, *this, device_memory, query_cache_runtime),
      pipeline_cache(device_memory, device, scheduler, descriptor_pool, guest_descriptor_queue,
                     descriptor_buffer_ring, render_pass_cache, buffer_cache, texture_cache,
                     gpu.ShaderNotify()),
      accelerate_dma(buffer_cache, texture_cache, scheduler),
      fence_manager(*this, gpu, texture_cache, buffer_cache, query_cache, device, scheduler),
      wfi_event(device.GetLogical().CreateEvent()) {
    scheduler.SetQueryCache(query_cache);
}

RasterizerVulkan::~RasterizerVulkan() {
    scheduler.WaitWorker();
    scheduler.Finish();
}

template <typename Func>
void RasterizerVulkan::PrepareDraw(bool is_indexed, Func&& draw_func) {

    SCOPE_EXIT {
        gpu.TickWork();
    };
    FlushWork();
    gpu_memory->FlushCaching();

    GraphicsPipeline* const pipeline{pipeline_cache.CurrentGraphicsPipeline()};
    if (!pipeline) {
        return;
    }
    std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
    // update engine as channel may be different.
    pipeline->SetEngine(maxwell3d, gpu_memory);
    if (!pipeline->Configure(is_indexed))
        return;

    UpdateDynamicStates();

    query_cache.NotifySegment(true);
    HandleTransformFeedback();
    query_cache.CounterEnable(VideoCommon::QueryType::ZPassPixelCount64, maxwell3d->regs.zpass_pixel_count_enable);
    draw_func();
}

void RasterizerVulkan::Draw(bool is_indexed, u32 instance_count) {
    PrepareDraw(is_indexed, [this, is_indexed, instance_count] {
        const auto& draw_state = maxwell3d->draw_manager.draw_state;
        const u32 num_instances{instance_count};
        const DrawParams draw_params{MakeDrawParams(draw_state, num_instances, is_indexed)};

        scheduler.Record([draw_params](vk::CommandBuffer cmdbuf) {
            if (draw_params.is_indexed) {
                cmdbuf.DrawIndexed(draw_params.num_vertices, draw_params.num_instances,
                                   draw_params.first_index, draw_params.base_vertex,
                                   draw_params.base_instance);
            } else {
                cmdbuf.Draw(draw_params.num_vertices, draw_params.num_instances,
                            draw_params.base_vertex, draw_params.base_instance);
            }
        });

        // Log draw call
        if (GPU::Logging::IsActive() &&
            Settings::values.gpu_log_vulkan_calls.GetValue()) {
            const std::string params = is_indexed ?
                fmt::format("vertices={}, instances={}, firstIndex={}, baseVertex={}, baseInstance={}",
                    draw_params.num_vertices, draw_params.num_instances,
                    draw_params.first_index, draw_params.base_vertex, draw_params.base_instance) :
                fmt::format("vertices={}, instances={}, firstVertex={}, firstInstance={}",
                    draw_params.num_vertices, draw_params.num_instances,
                    draw_params.base_vertex, draw_params.base_instance);
            GPU::Logging::GPULogger::GetInstance().LogVulkanCall(
                is_indexed ? "vkCmdDrawIndexed" : "vkCmdDraw", params, VK_SUCCESS);
        }
    });
}

void RasterizerVulkan::DrawIndirect() {
    const auto& params = maxwell3d->draw_manager.indirect_state;
    buffer_cache.SetDrawIndirect(&params);
    PrepareDraw(params.is_indexed, [this, &params] {
        const auto indirect_buffer = buffer_cache.GetDrawIndirectBuffer();
        const auto& buffer = indirect_buffer.first;
        const auto& offset = indirect_buffer.second;
        if (params.is_byte_count) {
            scheduler.Record([buffer_obj = buffer->Handle(), offset,
                              stride = params.stride](vk::CommandBuffer cmdbuf) {
                cmdbuf.DrawIndirectByteCountEXT(1, 0, buffer_obj, offset, 0,
                                                static_cast<u32>(stride));
            });
            return;
        }
        if (params.include_count) {
            const auto count = buffer_cache.GetDrawIndirectCount();
            const auto& draw_buffer = count.first;
            const auto& offset_base = count.second;
            scheduler.Record([draw_buffer_obj = draw_buffer->Handle(),
                              buffer_obj = buffer->Handle(), offset_base, offset,
                              params](vk::CommandBuffer cmdbuf) {
                if (params.is_indexed) {
                    cmdbuf.DrawIndexedIndirectCount(
                        buffer_obj, offset, draw_buffer_obj, offset_base,
                        static_cast<u32>(params.max_draw_counts), static_cast<u32>(params.stride));
                } else {
                    cmdbuf.DrawIndirectCount(buffer_obj, offset, draw_buffer_obj, offset_base,
                                             static_cast<u32>(params.max_draw_counts),
                                             static_cast<u32>(params.stride));
                }
            });
            return;
        }
        scheduler.Record([buffer_obj = buffer->Handle(), offset, params](vk::CommandBuffer cmdbuf) {
            if (params.is_indexed) {
                cmdbuf.DrawIndexedIndirect(buffer_obj, offset,
                                           static_cast<u32>(params.max_draw_counts),
                                           static_cast<u32>(params.stride));
            } else {
                cmdbuf.DrawIndirect(buffer_obj, offset, static_cast<u32>(params.max_draw_counts),
                                    static_cast<u32>(params.stride));
            }
        });

        // Log indirect draw call
        if (GPU::Logging::IsActive() &&
            Settings::values.gpu_log_vulkan_calls.GetValue()) {
            const std::string log_params = fmt::format("drawCount={}, stride={}",
                params.max_draw_counts, params.stride);
            GPU::Logging::GPULogger::GetInstance().LogVulkanCall(
                params.is_indexed ? "vkCmdDrawIndexedIndirect" : "vkCmdDrawIndirect",
                log_params, VK_SUCCESS);
        }
    });
    buffer_cache.SetDrawIndirect(nullptr);
}

void RasterizerVulkan::DrawTexture() {

    SCOPE_EXIT {
        gpu.TickWork();
    };
    FlushWork();

    std::scoped_lock l{texture_cache.mutex};
    texture_cache.SynchronizeDescriptors(false);
    texture_cache.UpdateRenderTargets(false);

    UpdateDynamicStates();

    query_cache.NotifySegment(true);
    query_cache.CounterEnable(VideoCommon::QueryType::ZPassPixelCount64, maxwell3d->regs.zpass_pixel_count_enable);
    const auto& draw_texture_state = maxwell3d->draw_manager.draw_texture_state;
    const auto& sampler = texture_cache.GetSampler(draw_texture_state.src_sampler, false);
    const auto& texture = texture_cache.GetImageView(draw_texture_state.src_texture);
    const auto* framebuffer = texture_cache.GetFramebuffer();

    const bool src_rescaling = texture_cache.IsRescaling() && texture.IsRescaled();
    const bool dst_rescaling = texture_cache.IsRescaling() && framebuffer->IsRescaled();

    const auto ScaleSrc = [&](auto dim_f) -> s32 {
        auto dim = static_cast<s32>(dim_f);
        return src_rescaling ? Settings::values.resolution_info.ScaleUp(dim) : dim;
    };

    const auto ScaleDst = [&](auto dim_f) -> s32 {
        auto dim = static_cast<s32>(dim_f);
        return dst_rescaling ? Settings::values.resolution_info.ScaleUp(dim) : dim;
    };

    Region2D dst_region = {Offset2D{.x = ScaleDst(draw_texture_state.dst_x0),
                                    .y = ScaleDst(draw_texture_state.dst_y0)},
                           Offset2D{.x = ScaleDst(draw_texture_state.dst_x1),
                                    .y = ScaleDst(draw_texture_state.dst_y1)}};
    Region2D src_region = {Offset2D{.x = ScaleSrc(draw_texture_state.src_x0),
                                    .y = ScaleSrc(draw_texture_state.src_y0)},
                           Offset2D{.x = ScaleSrc(draw_texture_state.src_x1),
                                    .y = ScaleSrc(draw_texture_state.src_y1)}};
    Extent3D src_size = {static_cast<u32>(ScaleSrc(texture.size.width)),
                         static_cast<u32>(ScaleSrc(texture.size.height)), texture.size.depth};
    blit_image.BlitColor(framebuffer, texture.RenderTarget(), texture.ImageHandle(),
                         sampler->Handle(), dst_region, src_region, src_size);
}

bool RasterizerVulkan::ClearNativeColor(const Tegra::Control::NativeColorClear& clear,
                                        std::stop_token stop_token) {
    const auto reject = [&](std::string_view reason) {
        LOG_ERROR(HW_GPU, "Native color clear rejected: {}, address={:#x}, extent={}x{}", reason,
                  clear.address, clear.width, clear.height);
        return false;
    };
    if (stop_token.stop_requested() || !gpu_memory)
        return reject("cancelled or missing address space");
    if (clear.Empty())
        return true;
    VideoCore::Surface::PixelFormat format;
    VkFormat host_format;
    switch (clear.format) {
    case Tegra::Control::NativeColorClear::Format::B10G11R11Float:
        format = VideoCore::Surface::PixelFormat::B10G11R11_FLOAT;
        host_format = VK_FORMAT_B10G11R11_UFLOAT_PACK32;
        break;
    case Tegra::Control::NativeColorClear::Format::RGBA8Srgb:
        format = VideoCore::Surface::PixelFormat::A8B8G8R8_SRGB;
        host_format = VK_FORMAT_A8B8G8R8_SRGB_PACK32;
        break;
    case Tegra::Control::NativeColorClear::Format::RGBA8Unorm:
        format = VideoCore::Surface::PixelFormat::A8B8G8R8_UNORM;
        host_format = VK_FORMAT_A8B8G8R8_UNORM_PACK32;
        break;
    default:
        return reject("unsupported target format");
    }
    const auto image_format =
        MaxwellToVK::SurfaceFormat(device, FormatType::Optimal, false, format);
    const auto view_format = MaxwellToVK::SurfaceFormat(device, FormatType::Optimal, true, format);
    VkFormatFeatureFlags required_features =
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
        VK_FORMAT_FEATURE_TRANSFER_DST_BIT | VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT;
    if (image_format.storage)
        required_features |= VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
    // Native staging has the guest's exact byte representation. A legacy format
    // fallback (e.g. packed float to RGBA16F) requires an explicit converter.
    if (!image_format.attachable || image_format.format != host_format ||
        view_format.format != host_format ||
        !device.IsFormatSupported(host_format, required_features, FormatType::Optimal)) {
        LOG_DEBUG(HW_GPU,
                  "Native clear host format: expected={}, image={}, view={}, attachable={}, "
                  "features={:#x}",
                  static_cast<u32>(host_format), static_cast<u32>(image_format.format),
                  static_cast<u32>(view_format.format), image_format.attachable, required_features);
        return reject("host format unavailable or substituted");
    }
    const auto limit = device.GetPhysical().GetProperties().limits.maxImageDimension2D;
    if (clear.width > limit || clear.height > limit)
        return reject("host image extent limit");
    FlushWork();
    // A target may reuse an allocation with a different format. Retire its old
    // image and views only after all previous host reads and writes complete.
    scheduler.Finish();
    gpu_memory->FlushCaching();
    auto lease = gpu_memory->AcquireMappingLease();
    const auto backing = lease.ResolveContiguous(clear.address, clear.size_bytes, true);
    if (!backing || stop_token.stop_requested())
        return reject("target backing or cancellation");
    {
        std::scoped_lock lock{buffer_cache.mutex};
        // Partial clears preserve the previous pixels. The initial native image
        // adapter cannot merge dirty SSBO contents into a cached image yet.
        if (buffer_cache.IsRegionGpuModified(*backing, clear.size_bytes)) {
            return reject("target overlaps pending buffer writes");
        }
    }
    VideoCommon::ImageInfo info;
    info.format = format;
    info.type = VideoCommon::ImageType::e2D;
    info.size = {clear.width, clear.height, 1};
    info.block = {0, clear.block_height, 0};
    info.layer_stride = clear.size_bytes;
    info.maybe_unaligned_layer_stride = clear.size_bytes;
    info.layout = Tegra::Texture::BlockLinearLayout::TuringColor2D;
    if (VideoCommon::CalculateGuestSizeInBytes(info) != clear.size_bytes)
        return reject("target footprint mismatch");
    const auto record_clear = [&] {
        std::scoped_lock lock{texture_cache.mutex};
        const auto target = texture_cache.PrepareNativeClearTarget(info, clear.address, *backing,
                                                                   false, false, true);
        if (!target || stop_token.stop_requested() ||
            target->framebuffer->RenderArea().width != clear.width ||
            target->framebuffer->RenderArea().height != clear.height ||
            target->framebuffer->NumColorBuffers() != 1 ||
            !target->framebuffer->HasAspectColorBit(0))
            return reject("cached target preparation, extent or cancellation");
        VkClearValue value{};
        std::memcpy(value.color.uint32, clear.color.data(), sizeof(value.color.uint32));
        const VkClearRect rect{
            .rect = {{static_cast<s32>(clear.x), static_cast<s32>(clear.y)},
                     {clear.extent_x, clear.extent_y}},
            .baseArrayLayer = 0,
            .layerCount = 1,
        };
        scheduler.RequestRenderpass(target->framebuffer);
        scheduler.Record([value, rect](vk::CommandBuffer cmdbuf) {
            const VkClearAttachment attachment{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .colorAttachment = 0,
                .clearValue = value,
            };
            cmdbuf.ClearAttachments(attachment, rect);
        });
        texture_cache.MarkModification(target->image_id);
        return true;
    };
    try {
        const bool recorded = record_clear();
        // Preparation can queue uploads even when the clear fails. Drain them
        // outside the cache lock, retaining the mapping lease on every exit.
        scheduler.Finish();
        if (!recorded)
            return false;
        {
            std::scoped_lock lock{texture_cache.mutex};
            // Publish the accepted clear to guest backing before future native
            // SSBO reads. The exact-image check bounds this download to the lease.
            texture_cache.DownloadMemory(*backing, clear.size_bytes);
        }
        {
            std::scoped_lock lock{buffer_cache.mutex};
            buffer_cache.WriteMemory(*backing, clear.size_bytes);
        }
    } catch (...) {
        try {
            scheduler.Finish();
        } catch (const std::exception& error) {
            LOG_CRITICAL(Render_Vulkan, "Native clear prefix drain failed: {}", error.what());
        }
        throw;
    }
    LOG_DEBUG(HW_GPU, "Native color clear completed: address={:#x}, extent={}x{}, rect={},{}/{},{}",
              clear.address, clear.width, clear.height, clear.x, clear.y, clear.extent_x,
              clear.extent_y);
    return !stop_token.stop_requested();
}

void RasterizerVulkan::Clear(u32 layer_count) {
    FlushWork();
    gpu_memory->FlushCaching();

    auto& regs = maxwell3d->regs;
    const bool use_color = regs.clear_surface.R || regs.clear_surface.G || regs.clear_surface.B ||
                           regs.clear_surface.A;
    const bool use_depth = regs.clear_surface.Z;
    const bool use_stencil = regs.clear_surface.S;
    if (!use_color && !use_depth && !use_stencil) {
        return;
    }

    std::scoped_lock lock{texture_cache.mutex};
    texture_cache.UpdateRenderTargets(true);
    const Framebuffer* const framebuffer = texture_cache.GetFramebuffer();
    const VkExtent2D render_area = framebuffer->RenderArea();

    constexpr bool ENABLE_DEFERRED_CLEAR = true;
    const bool color_full_channels = regs.clear_surface.R && regs.clear_surface.G &&
                                     regs.clear_surface.B && regs.clear_surface.A;
    const bool stencil_partial = use_stencil && framebuffer->HasAspectStencilBit() &&
                                 regs.stencil_front_mask != 0xFF && regs.stencil_front_mask != 0;
    const bool ds_used = use_depth || use_stencil;
    const bool ds_deferrable =
        !ds_used || ((!framebuffer->HasAspectDepthBit() || use_depth) &&
                     (!framebuffer->HasAspectStencilBit() || use_stencil) && !stencil_partial);
    u32 up_scale = 1;
    u32 down_shift = 0;
    if (texture_cache.IsRescaling()) {
        up_scale = Settings::values.resolution_info.up_scale;
        down_shift = Settings::values.resolution_info.down_shift;
    }
    VkRect2D default_scissor{};
    default_scissor.offset.x = 0;
    default_scissor.offset.y = 0;
    default_scissor.extent.width = (std::numeric_limits<s32>::max)();
    default_scissor.extent.height = (std::numeric_limits<s32>::max)();

    VkClearRect clear_rect{
        .rect = regs.clear_control.use_scissor ? GetScissorState(regs, 0, up_scale, down_shift)
                                               : default_scissor,
        .baseArrayLayer = regs.clear_surface.layer,
        .layerCount = layer_count,
    };
    const auto clamp_rect_to_render_area = [render_area](VkRect2D& rect) -> bool {
        const auto clamp_axis = [](s32& offset, u32& extent, u32 limit) {
            auto clamp_offset = [&offset, limit]() {
                if (limit == 0) {
                    offset = 0;
                    return;
                }
                offset = std::clamp(offset, 0, static_cast<s32>(limit));
            };

            if (extent == 0) {
                clamp_offset();
                return;
            }
            if (offset < 0) {
                const u32 shrink = (std::min)(extent, static_cast<u32>(-offset));
                extent -= shrink;
                offset = 0;
            }
            if (limit == 0) {
                extent = 0;
                offset = 0;
                return;
            }
            if (offset >= s32(limit)) {
                offset = s32(limit);
                extent = 0;
                return;
            }
            const u64 end_coord = u64(offset) + extent;
            if (end_coord > limit) {
                extent = limit - u32(offset);
            }
        };

        clamp_axis(rect.offset.x, rect.extent.width, render_area.width);
        clamp_axis(rect.offset.y, rect.extent.height, render_area.height);
        return rect.extent.width != 0 && rect.extent.height != 0;
    };
    if (!clamp_rect_to_render_area(clear_rect.rect)) {
        return;
    }

    const bool clear_covers_render_area =
        clear_rect.rect.offset.x == 0 && clear_rect.rect.offset.y == 0 &&
        clear_rect.rect.extent.width >= render_area.width &&
        clear_rect.rect.extent.height >= render_area.height;
    const bool can_defer_clear = ENABLE_DEFERRED_CLEAR && (!regs.clear_control.use_scissor || clear_covers_render_area) &&
                                 regs.clear_surface.layer == 0 &&
                                 !scheduler.IsRenderPassActive() &&
                                 (!use_color || color_full_channels) && ds_deferrable;
    if (!can_defer_clear) {
        scheduler.RequestRenderpass(framebuffer);
    }

    query_cache.NotifySegment(true);
    query_cache.CounterEnable(VideoCommon::QueryType::ZPassPixelCount64, maxwell3d->regs.zpass_pixel_count_enable);
    UpdateViewportsState(regs);

    const u32 color_attachment = regs.clear_surface.RT;
    if (use_color && framebuffer->HasAspectColorBit(color_attachment)) {
        const auto format = VideoCore::Surface::PixelFormatFromRenderTargetFormat(regs.rt[color_attachment].format);
        bool is_integer = IsPixelFormatInteger(format);
        bool is_signed = IsPixelFormatSignedInteger(format);
        size_t int_size = PixelComponentSizeBitsInteger(format);
        VkClearValue clear_value{};
        if (!is_integer) {
            std::memcpy(clear_value.color.float32, regs.clear_color.data(), regs.clear_color.size() * sizeof(f32));
        } else if (!is_signed) {
            for (size_t i = 0; i < 4; i++)
                clear_value.color.uint32[i] = u32(f32(u64(int_size) << 1U) * regs.clear_color[i]);
        } else {
            for (size_t i = 0; i < 4; i++)
                clear_value.color.int32[i] = s32(f32(s64(int_size - 1) << 1) * (regs.clear_color[i] - 0.5f));
        }

        if (color_full_channels) {
            if (can_defer_clear) {
                scheduler.DeferColorClear(framebuffer, color_attachment, clear_value);
            } else {
                scheduler.Record([color_attachment, clear_value, clear_rect](vk::CommandBuffer cmdbuf) {
                    const VkClearAttachment attachment{
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .colorAttachment = color_attachment,
                        .clearValue = clear_value,
                    };
                    cmdbuf.ClearAttachments(attachment, clear_rect);
                });
            }
        } else {
            u8 color_mask = u8(regs.clear_surface.R | regs.clear_surface.G << 1 | regs.clear_surface.B << 2 | regs.clear_surface.A << 3);
            Region2D dst_region = {
                Offset2D{.x = clear_rect.rect.offset.x, .y = clear_rect.rect.offset.y},
                Offset2D{.x = clear_rect.rect.offset.x + s32(clear_rect.rect.extent.width),
                         .y = clear_rect.rect.offset.y + s32(clear_rect.rect.extent.height)}};
            blit_image.ClearColor(framebuffer, color_mask, regs.clear_color, dst_region);
        }
    }

    if (!use_depth && !use_stencil) {
        return;
    }
    VkImageAspectFlags aspect_flags = 0;
    if (use_depth && framebuffer->HasAspectDepthBit()) {
        aspect_flags |= VK_IMAGE_ASPECT_DEPTH_BIT;
    }
    if (use_stencil && framebuffer->HasAspectStencilBit()) {
        aspect_flags |= VK_IMAGE_ASPECT_STENCIL_BIT;
    }
    if (aspect_flags == 0) {
        return;
    }

    if (use_stencil && framebuffer->HasAspectStencilBit() && regs.stencil_front_mask != 0xFF &&
        regs.stencil_front_mask != 0) {
        Region2D dst_region = {
            Offset2D{.x = clear_rect.rect.offset.x, .y = clear_rect.rect.offset.y},
            Offset2D{.x = clear_rect.rect.offset.x + s32(clear_rect.rect.extent.width),
                     .y = clear_rect.rect.offset.y + s32(clear_rect.rect.extent.height)}};
        blit_image.ClearDepthStencil(framebuffer, use_depth, regs.clear_depth,
                                     u8(regs.stencil_front_mask), regs.clear_stencil,
                                     regs.stencil_front_func_mask, dst_region);
    } else if (can_defer_clear) {
        VkClearValue ds_value{};
        ds_value.depthStencil.depth = regs.clear_depth;
        ds_value.depthStencil.stencil = regs.clear_stencil;
        scheduler.DeferDepthStencilClear(framebuffer, ds_value);
    } else {
        scheduler.Record([clear_depth = regs.clear_depth, clear_stencil = regs.clear_stencil,
                          clear_rect, aspect_flags](vk::CommandBuffer cmdbuf) {
            VkClearAttachment attachment;
            attachment.aspectMask = aspect_flags;
            attachment.colorAttachment = 0;
            attachment.clearValue.depthStencil.depth = clear_depth;
            attachment.clearValue.depthStencil.stencil = clear_stencil;
            cmdbuf.ClearAttachments(attachment, clear_rect);
        });
    }
}

void RasterizerVulkan::DispatchCompute() {
    DispatchComputeImpl(false);
}

bool RasterizerVulkan::DispatchNativeCompute() {
    return DispatchComputeImpl(true);
}

bool RasterizerVulkan::DispatchComputeImpl(bool require_bound_pipeline) {
    const bool native = kepler_compute->IsNativeMode();
    if ((require_bound_pipeline || native) &&
        !kepler_compute->IsNativeLaunchSupported(
            device.GetMaxComputeWorkGroupCount(), device.GetMaxComputeWorkGroupSize(),
            device.GetMaxComputeWorkGroupInvocations(), device.GetMaxComputeSharedMemorySize())) {
        return false;
    }
    FlushWork();
    gpu_memory->FlushCaching();

    const u64 mapping_generation = gpu_memory->MappingGeneration();
    ComputePipeline* const pipeline{pipeline_cache.CurrentComputePipeline()};
    if (!pipeline || ((require_bound_pipeline || native) && !pipeline->IsBound())) {
        return false;
    }
    const auto record_dispatch = [&](const ComputePipeline::NativeBindings* native_bindings) {
        if (native && kepler_compute->GetNativeStopToken().stop_requested()) {
            return false;
        }
        std::scoped_lock lock{texture_cache.mutex, buffer_cache.mutex};
        if (!pipeline->Configure(*kepler_compute, *gpu_memory, scheduler, buffer_cache,
                                 texture_cache, native_bindings)) {
            return false;
        }
        if (native && kepler_compute->GetNativeStopToken().stop_requested()) {
            return false;
        }

        const auto& qmd{kepler_compute->launch_description};
        auto indirect_address = kepler_compute->GetIndirectComputeAddress();
        if (indirect_address) {
            // DispatchIndirect
            static constexpr auto sync_info = VideoCommon::ObtainBufferSynchronize::FullSynchronize;
            const auto post_op = VideoCommon::ObtainBufferOperation::DiscardWrite;
            const auto [buffer, offset] =
                buffer_cache.ObtainBuffer(*indirect_address, 12, sync_info, post_op);
            scheduler.RequestOutsideRenderPassOperationContext();
            scheduler.Record([pipeline, indirect_buffer = buffer->Handle(),
                              indirect_offset = offset](vk::CommandBuffer cmdbuf) {
                if (!pipeline->IsBound()) {
                    return;
                }
                cmdbuf.DispatchIndirect(indirect_buffer, indirect_offset);
            });
            return true;
        }
        const std::array<u32, 3> dim{qmd.grid_dim_x, qmd.grid_dim_y, qmd.grid_dim_z};
        const std::array<u32, 3> max_dim{device.GetMaxComputeWorkGroupCount()};
        if (dim[0] > max_dim[0] || dim[1] > max_dim[1] || dim[2] > max_dim[2]) {
            return false;
        }
        scheduler.RequestOutsideRenderPassOperationContext();
        static constexpr VkMemoryBarrier READ_BARRIER{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT,
        };
        scheduler.Record([](vk::CommandBuffer cmdbuf) {
            cmdbuf.PipelineBarrier(vk::PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER,
                                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, READ_BARRIER);
        });
        scheduler.Record([pipeline, dim](vk::CommandBuffer cmdbuf) {
            if (!pipeline->IsBound()) {
                return;
            }
            cmdbuf.Dispatch(dim[0], dim[1], dim[2]);
        });

        // Log compute dispatch
        if (GPU::Logging::IsActive() && Settings::values.gpu_log_vulkan_calls.GetValue()) {
            const std::string params = fmt::format("groupCountX={}, groupCountY={}, groupCountZ={}",
                                                   dim[0], dim[1], dim[2]);
            GPU::Logging::GPULogger::GetInstance().LogVulkanCall("vkCmdDispatch", params,
                                                                 VK_SUCCESS);
        }
        return true;
    };
    if (!native) {
        return record_dispatch(nullptr);
    }
    // Checked descriptor reads may flush cache-backed memory. Snapshot them before
    // taking the mapping lease or renderer locks; never reread under those locks.
    auto bindings = pipeline->SnapshotNativeBindings(*kepler_compute);
    if (!bindings) {
        return false;
    }
    auto lease = gpu_memory->AcquireMappingLease();
    if (gpu_memory->MappingGeneration() != mapping_generation ||
        !pipeline->ResolveNativeBindings(*bindings, lease, kepler_compute->GetNativeStopToken())) {
        return false;
    }
    try {
        const bool recorded = record_dispatch(&*bindings);
        // The cache locks have been released. Unmap/free must wait until actual GPU
        // completion, including any uploads recorded by a partially failed Configure.
        scheduler.Finish();
        return recorded;
    } catch (...) {
        try {
            scheduler.Finish();
        } catch (const std::exception& error) {
            LOG_CRITICAL(Render_Vulkan, "Native compute prefix drain failed: {}", error.what());
        }
        throw;
    }
}

void RasterizerVulkan::ResetCounter(VideoCommon::QueryType type) {
    switch (type) {
    case VideoCommon::QueryType::ZPassPixelCount64:
    case VideoCommon::QueryType::StreamingByteCount:
    case VideoCommon::QueryType::StreamingPrimitivesSucceeded:
    case VideoCommon::QueryType::VtgPrimitivesOut:
        query_cache.CounterReset(type);
        return;
    default:
        LOG_DEBUG(Render_Vulkan, "Unimplemented counter reset={}", type);
        return;
    }
}

bool RasterizerVulkan::SynchronizeNativeWork() {
    scheduler.Finish();
    return true;
}

bool RasterizerVulkan::DrawNativeIndexed(const Tegra::Control::NativeIndexedDrawRequest& request,
                                         std::stop_token stop_token) {
    using Request = Tegra::Control::NativeIndexedDrawRequest;
    using Bounds = Tegra::Control::NativeIndexBounds;
    using Fetch = Tegra::Control::NativeVertexFetch;
    using Clear = Tegra::Control::NativeColorClear;
    const auto reject = [](std::string_view reason) {
        LOG_ERROR(HW_GPU, "Native Vulkan draw rejected: {}", reason);
        return false;
    };
    if (!maxwell3d || !gpu_memory || stop_token.stop_requested())
        return false;
    const u32 index_format = static_cast<u32>(request.format);
    constexpr u64 SnapshotLimit = 64ULL << 20;
    if (index_format > 2 || request.size_bytes != (u64{request.count} << index_format) ||
        request.address >= (u64{1} << 40) ||
        request.size_bytes > (u64{1} << 40) - request.address || request.size_bytes > SnapshotLimit)
        return reject("invalid or unsupported index snapshot size");
    FlushWork();
    gpu_memory->FlushCaching();
    scheduler.Finish();
    if (!request.count)
        return !stop_token.stop_requested();
    if (device.IsMoltenVK())
        return reject("native triangle-list restart disable requires host support");
    if (!device.SupportsMultiViewport() && device.GetMaxViewports() > 1)
        return reject("native static viewport count requires host support");

    const auto read = [&](u32 method) { return maxwell3d->GetRegisterValue(method); };
    // The initial adapter models a single-sample, single-color triangle draw.
    // Reject effects that have no native resource/state adapter yet.
    bool unsupported_state{};
    for (u32 method :
         {0x1d0U, 0x1d1U, 0x370U, 0x371U, 0x372U, 0x374U, 0x3d0U, 0x3e9U, 0x452U, 0x47eU, 0x4bbU,
          0x4d5U, 0x4e0U, 0x54dU, 0x54eU, 0x54fU, 0x59bU, 0x653U, 0x65fU, 0x66fU, 0x671U}) {
        if (read(method)) {
            LOG_ERROR(HW_GPU, "Unsupported native draw state: method={:#x}, value={:#x}", method,
                      read(method));
            unsupported_state = true;
        }
    }
    if (unsupported_state)
        return false;
    // ZT_SELECT is zero above: there is no depth attachment, so the stored
    // depth-test request cannot enable testing on this color-only framebuffer.
    if (read(0x4b3) > 1)
        return reject("invalid native depth-test enable");
    if (read(0x545) > 1 || (read(0x545) && !Settings::IsGPULevelHigh()))
        return reject("native pixel counting requires precise GPU queries");
    // ZCULL_STATS enables internal counters without changing rasterization.
    // Keep the stored enable state; native stats reports/resets remain rejected
    // until their distinct counters can be materialized (not substituted by Z-pass).
    if (read(0x547) > 1)
        return reject("invalid native Z-cull statistics enable");
    if (read(0xdf) > 1 || read(0x35f) > 1 || read(0x4eb) != 0 || read(0x649) != 0 ||
        read(0x64b) != 1 || read(0x652) != 0 || read(0x54f) != 0 || read(0x3ee) != 0 ||
        read(0x574) != 0 || read(0x646) > 1 || read(0x4d7) > 1 || read(0x36b) != 0x1b02 ||
        read(0x36c) != 0x1b02)
        return reject("unsupported raster, viewport or multisample mode");
    const u32 clip = read(0x64f);
    const auto depth_clip =
        Tegra::Control::NativeDepthClip::Decode(clip, read(0x35f), read(0x302), read(0x303));
    if (!depth_clip) {
        LOG_ERROR(HW_GPU, "Unsupported native clipping policy: clip={:#x}, z_range={:#x}", clip,
                  read(0x35f));
        return false;
    }
    if (read(0x800) & 1 || read(0x810) != 0x11 || read(0x850) != 0x51 || (read(0x820) & 1) ||
        (read(0x830) & 1) || (read(0x840) & 1))
        return reject("unsupported native shader-stage combination");
    Tegra::Control::NativeRenderTargets::Registers target_registers{};
    for (u32 index = 0; index < target_registers.size(); ++index)
        for (u32 word = 0; word < target_registers[index].size(); ++word)
            target_registers[index][word] = read(0x200 + index * 0x10 + word);
    const auto targets =
        Tegra::Control::NativeRenderTargets::Decode(read(0x487), read(0x56e), target_registers);
    if (!targets)
        return reject("unsupported native color target or sRGB write mode");
    if (targets->count > 1 && read(0x4d7))
        return reject("unsupported native shared ROP control");
    const auto* target = &targets->targets[0].color;
    if (read(0x980) > 1 || (target->format == Clear::Format::B10G11R11Float && read(0x980)))
        return reject("unsupported native floating-point color clamp");
    VideoCore::Surface::PixelFormat format;
    VkFormat host_format;
    switch (target->format) {
    case Clear::Format::B10G11R11Float:
        format = VideoCore::Surface::PixelFormat::B10G11R11_FLOAT;
        host_format = VK_FORMAT_B10G11R11_UFLOAT_PACK32;
        break;
    case Clear::Format::RGBA8Srgb:
        format = VideoCore::Surface::PixelFormat::A8B8G8R8_SRGB;
        host_format = VK_FORMAT_A8B8G8R8_SRGB_PACK32;
        break;
    case Clear::Format::RGBA8Unorm:
        format = VideoCore::Surface::PixelFormat::A8B8G8R8_UNORM;
        host_format = VK_FORMAT_A8B8G8R8_UNORM_PACK32;
        break;
    default:
        return reject("unsupported native color target format");
    }
    const auto surface_format =
        MaxwellToVK::SurfaceFormat(device, FormatType::Optimal, false, format);
    const auto view_format = MaxwellToVK::SurfaceFormat(device, FormatType::Optimal, true, format);
    VkFormatFeatureFlags required =
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
        VK_FORMAT_FEATURE_TRANSFER_DST_BIT | VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT;
    if (surface_format.storage)
        required |= VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
    const auto limits = device.GetPhysical().GetProperties().limits;
    if (!surface_format.attachable || surface_format.format != host_format ||
        view_format.format != host_format ||
        !device.IsFormatSupported(host_format, required, FormatType::Optimal) ||
        target->width > limits.maxImageDimension2D || target->height > limits.maxImageDimension2D ||
        targets->count > limits.maxColorAttachments)
        return reject("native target format or dimensions unavailable on host");

    const auto& regs = maxwell3d->regs;
    const auto& transform = regs.viewport_transform[0];
    const std::array floats{transform.scale_x,     transform.scale_y,     transform.scale_z,
                            transform.translate_x, transform.translate_y, transform.translate_z};
    if (!std::ranges::all_of(floats, [](float value) { return std::isfinite(value); }) ||
        transform.scale_x <= 0 || transform.scale_y == 0 || transform.swizzle.raw != 0x6420)
        return reject("unsupported native viewport transform");
    const auto viewport = GetViewportState(device, regs, 0, 1.f);
    const float x1 = viewport.x + viewport.width;
    const float y1 = viewport.y + viewport.height;
    if (!std::isfinite(x1) || !std::isfinite(y1) ||
        viewport.width > limits.maxViewportDimensions[0] ||
        std::abs(viewport.height) > limits.maxViewportDimensions[1] ||
        (std::min)(viewport.x, x1) < limits.viewportBoundsRange[0] ||
        (std::max)(viewport.x, x1) > limits.viewportBoundsRange[1] ||
        (std::min)(viewport.y, y1) < limits.viewportBoundsRange[0] ||
        (std::max)(viewport.y, y1) > limits.viewportBoundsRange[1])
        return reject("native viewport exceeds host limits");
    const float min_depth = transform.translate_z - (read(0x35f) == 0 ? transform.scale_z : 0.f);
    const float max_depth = transform.translate_z + transform.scale_z;
    if (min_depth != 0.f || max_depth != 1.f)
        return reject("native pixel depth clamps require the supported viewport depth range");
    if (!std::isfinite(min_depth) || !std::isfinite(max_depth))
        return reject("non-finite native viewport depth range");
    if (!device.IsExtDepthRangeUnrestrictedSupported() &&
        (min_depth < 0 || min_depth > 1 || max_depth < 0 || max_depth > 1))
        return reject("native depth range requires unrestricted host support");
    u32 x0{}, y0{}, x2 = target->width, y2 = target->height;
    const auto intersect = [&](u32 horizontal, u32 vertical, bool origin_extent) {
        const u32 left = horizontal & 0xffff, top = vertical & 0xffff;
        const u32 right = origin_extent ? left + (horizontal >> 16) : horizontal >> 16;
        const u32 bottom = origin_extent ? top + (vertical >> 16) : vertical >> 16;
        x0 = (std::max)(x0, (std::min)(left, target->width));
        y0 = (std::max)(y0, (std::min)(top, target->height));
        x2 = (std::min)(x2, right);
        y2 = (std::min)(y2, bottom);
    };
    intersect(read(0x3fd), read(0x3fe), true);
    intersect(read(0x300), read(0x301), true);
    if (read(0x380) > 1)
        return reject("invalid native scissor enable");
    if (read(0x380))
        intersect(read(0x381), read(0x382), false);
    const VkRect2D scissor{{static_cast<s32>(x0), static_cast<s32>(y0)},
                           {x2 > x0 ? x2 - x0 : 0, y2 > y0 ? y2 - y0 : 0}};

    auto lease = gpu_memory->AcquireMappingLease();
    // Failure drains must precede destruction of owned sampled resources.
    std::vector<std::unique_ptr<Image>> native_sample_images;
    std::vector<vk::ImageView> native_sample_views;
    std::vector<vk::Sampler> native_samplers;
    u64 native_sample_copy_bytes{};
    bool work_drained{};
    SCOPE_EXIT {
        if (!work_drained) {
            try {
                scheduler.Finish();
            } catch (const std::exception& error) {
                LOG_CRITICAL(Render_Vulkan, "Native draw prefix drain failed: {}", error.what());
            }
        }
    };
    std::array<DAddr, 8> outputs{};
    u64 output_bytes{};
    for (u32 index = 0; index < targets->count; ++index) {
        const auto& attachment = targets->targets[index];
        if (attachment.color.size_bytes > SnapshotLimit - output_bytes)
            return reject("native target set exceeds snapshot capacity");
        output_bytes += attachment.color.size_bytes;
        // This view touches only its selected window; padded Z strides may
        // overlap siblings or extend beyond their allocation and are never read.
        const auto backing =
            lease.ResolveContiguous(attachment.color.address, attachment.color.size_bytes, true);
        if (!backing)
            return reject("unbacked native color target");
        if (attachment.single_gob_volume &&
            (gpu_memory->GetPageKind(attachment.color.address) != Tegra::PTEKind::GENERIC_16BX2 ||
             gpu_memory->GetMemoryLayoutSize(attachment.color.address,
                                             attachment.color.size_bytes) <
                 attachment.color.size_bytes))
            return reject("unsupported native color-slice memory kind");
        outputs[index] = *backing;
        for (u32 previous = 0; previous < index; ++previous)
            if (outputs[index] < outputs[previous] + targets->targets[previous].color.size_bytes &&
                outputs[previous] < outputs[index] + attachment.color.size_bytes)
                return reject("native color targets alias physical backing");
    }
    const auto overlaps_output = [&](DAddr address, u64 size) {
        for (u32 index = 0; index < targets->count; ++index)
            if (address < outputs[index] + targets->targets[index].color.size_bytes &&
                outputs[index] < address + size)
                return true;
        return false;
    };
    {
        std::scoped_lock lock{buffer_cache.mutex};
        for (u32 index = 0; index < targets->count; ++index)
            if (buffer_cache.IsRegionGpuModified(outputs[index],
                                                 targets->targets[index].color.size_bytes))
                return reject("native target overlaps pending buffer writes");
    }
    u64 snapshot_bytes{};
    const auto snapshot = [&](GPUVAddr address, size_t size, std::vector<u8>& bytes) {
        if (!size || size > SnapshotLimit - snapshot_bytes || stop_token.stop_requested())
            return false;
        const auto backing = lease.ResolveContiguous(address, size, false);
        if (!backing || overlaps_output(*backing, size))
            return false;
        {
            std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
            if (buffer_cache.IsRegionGpuModified(*backing, size) ||
                texture_cache.IsRegionGpuModified(*backing, size))
                return false;
        }
        bytes.resize(size);
        if (!lease.ReadBlockChecked(address, bytes))
            return false;
        snapshot_bytes += size;
        return true;
    };
    std::vector<u8> indices;
    if (!snapshot(request.address, static_cast<size_t>(request.size_bytes), indices))
        return reject("native index backing or coherence");
    const auto bounds = Bounds::Decode(request.format, indices, stop_token);
    if (!bounds)
        return reject("invalid native index data");
    VkIndexType host_index_format = index_format == 2 ? VK_INDEX_TYPE_UINT32 : VK_INDEX_TYPE_UINT16;
    if (request.format == Request::Format::U8) {
        if (indices.size() > (SnapshotLimit - snapshot_bytes) / sizeof(u16))
            return reject("native index widening exceeds snapshot capacity");
        std::vector<u8> widened(indices.size() * sizeof(u16));
        for (size_t i = 0; i < indices.size(); ++i) {
            if ((i & 0xfff) == 0 && stop_token.stop_requested())
                return false;
            widened[2 * i] = indices[i];
            widened[2 * i + 1] = 0;
        }
        snapshot_bytes += widened.size();
        indices = std::move(widened);
    }
    const GPUVAddr vertex_header = (GPUVAddr{read(0x815)} << 32) | read(0x816);
    const GPUVAddr fragment_header = (GPUVAddr{read(0x855)} << 32) | read(0x856);
    VideoCommon::GraphicsEnvironment vertex{
        *maxwell3d, *gpu_memory, Maxwell::ShaderType::VertexB,
        VideoCommon::GraphicsEnvironment::NativeProgram{vertex_header, stop_token, &lease}};
    VideoCommon::GraphicsEnvironment fragment{
        *maxwell3d, *gpu_memory, Maxwell::ShaderType::Pixel,
        VideoCommon::GraphicsEnvironment::NativeProgram{fragment_header, stop_token, &lease}};
    GraphicsPipelineCacheKey key{};
    key.unique_hashes[1] = vertex.CalculateHash();
    key.unique_hashes[5] = fragment.CalculateHash();
    key.state.topology.Assign(Maxwell::PrimitiveTopology::Triangles);
    key.state.polygon_mode.Assign(FixedPipelineState::PackPolygonMode(Maxwell::PolygonMode::Fill));
    key.state.alpha_test_func.Assign(
        FixedPipelineState::PackComparisonOp(Maxwell::ComparisonOp::Always_GL));
    key.state.ndc_minus_one_to_one.Assign(depth_clip->negative_one_to_one);
    key.state.driver_id = device.GetDriverID();
    key.state.driver_version = device.GetDriverVersion();
    for (u32 index = 0; index < targets->count; ++index)
        key.state.color_formats[index] = static_cast<u8>(target_registers[index][4]);
    key.state.viewport_swizzles.fill(0x6420);
    key.state.dynamic_state.rasterize_enable.Assign(read(0xdf));
    key.state.dynamic_state.depth_clamp_disabled.Assign(1);
    if (read(0x5a1) > 1 || (read(0x5a1) && !device.SupportsProvokingVertexLastMode()))
        return reject("unsupported native provoking vertex mode");
    key.state.provoking_vertex_last.Assign(read(0x5a1));
    const u32 front = read(0x647), cull = read(0x648);
    if ((front != 0x900 && front != 0x901) || (cull != 0x404 && cull != 0x405 && cull != 0x408))
        return reject("unsupported native face mode");
    key.state.dynamic_state.front_face.Assign(
        FixedPipelineState::PackFrontFace(static_cast<Maxwell::FrontFace>(front)));
    key.state.dynamic_state.cull_enable.Assign(read(0x646));
    key.state.dynamic_state.cull_face.Assign(
        FixedPipelineState::PackCullFace(static_cast<Maxwell::CullFace>(cull)));
    if (read(0x4b9) > 1 || read(0x3e4) > 1)
        return reject("invalid native per-target blend enable");
    for (u32 index = 0; index < targets->count; ++index) {
        const u32 color_mask = read(0x680 + (read(0x3e4) ? 0 : index));
        if (color_mask & ~0x1111U)
            return reject("invalid native color write mask");
        const u32 blend_base = read(0x4b9) ? 0x780 + index * 8 : 0x4cf;
        const auto blend = FixedPipelineState::DecodeNativeBlend(
            {read(0x4d8 + index), read(blend_base), read(blend_base + 1), read(blend_base + 2),
             read(blend_base + 3), read(blend_base + 4), read(blend_base + 5),
             read(blend_base + (read(0x4b9) ? 6 : 7)), color_mask});
        if (!blend)
            return reject("unsupported native blend operation or factors");
        if (blend->enable &&
            !device.IsFormatSupported(host_format, VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT,
                                      FormatType::Optimal))
            return reject("native target blending unavailable on host");
        key.state.attachments[index] = *blend;
    }
    const std::array blend_constants{regs.blend_color.r, regs.blend_color.g, regs.blend_color.b,
                                     regs.blend_color.a};
    if (!std::ranges::all_of(blend_constants, [](float value) { return std::isfinite(value); }))
        return reject("unsupported non-finite native blend constants");
    std::array<std::vector<u8>, 32> vertices;
    for (u32 attribute = 0; attribute < 32; ++attribute) {
        const auto input = vertex.SPH().vtg.InputGeneric(attribute);
        if (std::ranges::none_of(input, [](bool used) { return used; }))
            continue;
        const u32 raw = read(0x458 + attribute), stream = raw & 31;
        const u32 stream_base = 0x700 + stream * 4;
        const Fetch::Registers fetch_regs{raw,
                                          read(stream_base),
                                          read(stream_base + 1),
                                          read(stream_base + 2),
                                          read(0x180 + stream * 2),
                                          read(0x181 + stream * 2),
                                          read(0x620 + stream)};
        const auto fetch = Fetch::Decode(fetch_regs, bounds->maximum);
        if (!fetch || fetch->size_bytes > SnapshotLimit ||
            attribute >= device.GetMaxVertexInputAttributes() ||
            stream >= device.GetMaxVertexInputBindings() ||
            fetch->stride > limits.maxVertexInputBindingStride ||
            fetch->offset > limits.maxVertexInputAttributeOffset)
            return reject("unsupported native vertex fetch range or format");
        VkFormat vertex_format{};
        switch (fetch->component_format) {
        case 1:
            vertex_format = VK_FORMAT_R32G32B32A32_SFLOAT;
            break;
        case 2:
            vertex_format = VK_FORMAT_R32G32B32_SFLOAT;
            break;
        case 3:
            vertex_format = VK_FORMAT_R16G16B16A16_SFLOAT;
            break;
        case 4:
            vertex_format = VK_FORMAT_R32G32_SFLOAT;
            break;
        case 5:
            vertex_format = VK_FORMAT_R16G16B16_SFLOAT;
            break;
        case 15:
            vertex_format = VK_FORMAT_R16G16_SFLOAT;
            break;
        case 18:
            vertex_format = VK_FORMAT_R32_SFLOAT;
            break;
        case 27:
            vertex_format = VK_FORMAT_R16_SFLOAT;
            break;
        default:
            return reject("unsupported native vertex component format");
        }
        if (!device.IsFormatSupported(vertex_format, VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT,
                                      FormatType::Buffer) ||
            MaxwellToVK::VertexFormat(device, Maxwell::VertexAttribute::Type::Float,
                                      static_cast<Maxwell::VertexAttribute::Size>(
                                          fetch->component_format)) != vertex_format)
            return reject("native vertex format unavailable on host");
        const u32 alignment = fetch->component_format == 1 || fetch->component_format == 2 ||
                                      fetch->component_format == 4 || fetch->component_format == 18
                                  ? 4
                                  : 2;
        if (fetch->offset % alignment || fetch->stride % alignment)
            return reject("native vertex format alignment requires repacking");
        auto& entry = key.state.attributes[attribute];
        entry.enabled.Assign(1);
        entry.buffer.Assign(stream);
        entry.offset.Assign(fetch->offset);
        entry.type.Assign(7);
        entry.size.Assign(fetch->component_format);
        key.state.vertex_strides[stream] = static_cast<u16>(fetch->stride);
        if (fetch->size_bytes > vertices[stream].size() &&
            !snapshot(fetch->address, static_cast<size_t>(fetch->size_bytes), vertices[stream]))
            return reject("native vertex backing or coherence");
    }
    const std::array environments{&vertex, &fragment};
    auto* pipeline = pipeline_cache.NativeGraphicsPipeline(key, environments);
    if (!pipeline || stop_token.stop_requested())
        return reject("native graphics pipeline construction");
    if (pipeline->NativeStageInfo(4).loads[Shader::IR::Attribute::PositionZ])
        return reject("native fragment depth reads require a pixel-clamp adapter");
    std::array<std::array<std::vector<u8>, 18>, 5> constants;
    std::vector<GraphicsPipeline::NativeUniformBuffer> uniforms;
    struct SampledRequest {
        u32 stage, index;
        GPUVAddr address;
        DAddr backing;
        VideoCommon::ImageInfo info;
        VkFormat host_format;
        VkSampler sampler;
        std::vector<u8> uploaded_pixels;
    };
    std::vector<SampledRequest> sampled_requests;
    for (u32 stage = 0; stage < 5; ++stage) {
        const auto& shader = pipeline->NativeStageInfo(stage);
        if (shader.uses_render_area && (!regs.surface_clip.width || !regs.surface_clip.height))
            return reject("empty native render-area specialization");
        for (const auto& descriptor : shader.constant_buffer_descriptors) {
            if (descriptor.index >= constants[stage].size() || descriptor.count != 1)
                return reject("unsupported native uniform descriptors");
            const auto& binding =
                maxwell3d->state.shader_stages[stage].const_buffers[descriptor.index];
            const u32 declared_size = shader.constant_buffer_used_sizes[descriptor.index];
            const u32 bound_size = shader.native_constant_buffer_sizes[descriptor.index];
            // Back the complete 16-byte packet used by hosts without scalar
            // UBO layouts, while reading only inside the actual guest binding.
            const u32 size = (std::min)(Common::AlignUp(declared_size, 16U), bound_size);
            if (!binding.enabled || !size || binding.size != bound_size ||
                size > limits.maxUniformBufferRange ||
                !snapshot(binding.address, size, constants[stage][descriptor.index])) {
                LOG_DEBUG(HW_GPU,
                          "Native uniform capture rejected: stage={}, bank={}, address={:#x}, "
                          "enabled={}, bound_bytes={:#x}, shader_bytes={:#x}, host_limit={:#x}",
                          stage, descriptor.index, binding.address, binding.enabled, binding.size,
                          declared_size, limits.maxUniformBufferRange);
                return reject("native uniform backing, bounds or coherence");
            }
            uniforms.push_back({stage, descriptor.index, constants[stage][descriptor.index]});
        }
        for (u32 i = 0; i < shader.texture_descriptors.size(); ++i) {
            if (stop_token.stop_requested())
                return false;
            if ((stage != 0 && stage != 4) || sampled_requests.size() >= 64)
                return reject("unsupported native texture stage or count");
            const auto& descriptor = shader.texture_descriptors[i];
            if (descriptor.count != 1 || descriptor.shift_left || descriptor.has_secondary ||
                descriptor.is_depth || descriptor.is_multisample || descriptor.is_integer ||
                (descriptor.type != Shader::TextureType::Color2D &&
                 descriptor.type != Shader::TextureType::Color3D))
                return reject("unsupported native sampled-image descriptor");
            auto& environment = stage == 0 ? vertex : fragment;
            const u32 handle =
                environment.ReadCbufValue(descriptor.cbuf_index, descriptor.cbuf_offset);
            const auto bindings = environment.NativeTextures();
            const auto binding = std::ranges::find(
                bindings, handle, &VideoCommon::GraphicsEnvironment::NativeTextureBinding::handle);
            if (binding == bindings.end())
                return reject("uncaptured native sampled-image handle");
            const auto mip =
                Tegra::Control::SelectNativeSampledMip(binding->texture, binding->sampler);
            if (!mip) {
                const auto& tic = binding->texture.words;
                const auto& tsc = binding->sampler.words;
                LOG_DEBUG(HW_GPU,
                          "Native sampled state rejected: handle={:#x}, "
                          "tic={:#x}/{:#x}/{:#x}/{:#x}/{:#x}/{:#x}/{:#x}/{:#x}, "
                          "tsc={:#x}/{:#x}/{:#x}/{:#x}/{:#x}/{:#x}/{:#x}/{:#x}",
                          handle, tic[0], tic[1], tic[2], tic[3], tic[4], tic[5], tic[6], tic[7],
                          tsc[0], tsc[1], tsc[2], tsc[3], tsc[4], tsc[5], tsc[6], tsc[7]);
                return reject("unsupported native fixed-mip sampling state");
            }
            const bool volume = binding->texture.IsB10G11R11Float3D();
            if (descriptor.type !=
                (volume ? Shader::TextureType::Color3D : Shader::TextureType::Color2D))
                return reject("native sampled-image dimensionality mismatch");
            const auto& sampler = binding->sampler;
            const u64 size = mip->size_bytes;
            const u32 extent_limit =
                volume ? limits.maxImageDimension3D : limits.maxImageDimension2D;
            if (!size || size > SnapshotLimit || mip->width > extent_limit ||
                mip->height > extent_limit || mip->depth > extent_limit)
                return reject("unsupported native sampled-image extent");
            const auto backing = lease.ResolveContiguous(mip->address, size, false);
            if (!backing || overlaps_output(*backing, size))
                return reject("native sampled-image backing or attachment alias");
            if (volume && (gpu_memory->GetPageKind(mip->address) != Tegra::PTEKind::GENERIC_16BX2 ||
                           gpu_memory->GetMemoryLayoutSize(mip->address, size) < size))
                return reject("unsupported native color-volume memory kind");
            {
                std::scoped_lock lock{buffer_cache.mutex};
                if (buffer_cache.IsRegionGpuModified(*backing, size))
                    return reject("native sampled-image overlaps pending buffer writes");
            }
            VkFormatFeatureFlags features = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
            if (sampler.mag_filter == 2 || sampler.min_filter == 2)
                features |= VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
            const bool packed_float = binding->texture.IsB10G11R11Float2D() || volume;
            const bool astc = binding->texture.IsAstc4x4Unorm2D();
            const bool bc4 = binding->texture.IsBc4UnormRrr1_2D();
            const VkFormat sampled_format = packed_float ? VK_FORMAT_B10G11R11_UFLOAT_PACK32
                                            : bc4        ? VK_FORMAT_R8_UNORM
                                            : binding->texture.srgb
                                                ? VK_FORMAT_A8B8G8R8_SRGB_PACK32
                                                : VK_FORMAT_A8B8G8R8_UNORM_PACK32;
            if (astc || bc4) {
                const auto decoded_format = bc4 ? VideoCore::Surface::PixelFormat::R8_UNORM
                                                : VideoCore::Surface::PixelFormat::A8B8G8R8_SRGB;
                const auto host_surface =
                    MaxwellToVK::SurfaceFormat(device, FormatType::Optimal, false, decoded_format);
                const auto host_view =
                    MaxwellToVK::SurfaceFormat(device, FormatType::Optimal, true, decoded_format);
                if (!host_surface.attachable || host_surface.format != sampled_format ||
                    host_view.format != sampled_format)
                    return reject("native decoded compressed host format unavailable");
                features |= VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
                            VK_FORMAT_FEATURE_TRANSFER_DST_BIT |
                            VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT;
                if (host_surface.storage)
                    features |= VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
            }
            if (!device.IsFormatSupported(sampled_format, features, FormatType::Optimal))
                return reject("native sampled-image filtering unavailable on host");
            VideoCommon::ImageInfo sampled;
            sampled.format = packed_float ? VideoCore::Surface::PixelFormat::B10G11R11_FLOAT
                             : astc       ? VideoCore::Surface::PixelFormat::ASTC_2D_4X4_SRGB
                             : bc4        ? VideoCore::Surface::PixelFormat::BC4_UNORM
                             : binding->texture.srgb
                                 ? VideoCore::Surface::PixelFormat::A8B8G8R8_SRGB
                                 : VideoCore::Surface::PixelFormat::A8B8G8R8_UNORM;
            sampled.type = volume ? VideoCommon::ImageType::e3D : VideoCommon::ImageType::e2D;
            sampled.size = {mip->width, mip->height, mip->depth};
            sampled.block = {0, mip->block_height, mip->block_depth};
            sampled.layer_stride = sampled.maybe_unaligned_layer_stride = static_cast<u32>(size);
            sampled.layout = Tegra::Texture::BlockLinearLayout::TuringColor2D;
            if (VideoCommon::CalculateGuestSizeInBytes(sampled) != size)
                return reject("native sampled-image footprint mismatch");
            std::vector<u8> uploaded_pixels;
            if (astc || bc4) {
                // Native C797 color blocklinear images use Turing's sector
                // order. Require the complete generic mapping,
                // not just an opaque descriptor preset or the first page kind.
                if (gpu_memory->GetPageKind(mip->address) != Tegra::PTEKind::GENERIC_16BX2 ||
                    gpu_memory->GetMemoryLayoutSize(mip->address, size) < size)
                    return reject("unsupported native compressed memory kind");
                std::vector<u8> tiled;
                if (!snapshot(mip->address, size, tiled))
                    return reject("native compressed backing or coherence");
                const u64 columns = (u64{mip->width} + 3) / 4;
                const u64 rows = (u64{mip->height} + 3) / 4;
                const u32 bytes_per_block = bc4 ? 8 : 16;
                const u64 linear_size = columns * rows * bytes_per_block;
                const u64 decoded_size = u64{mip->width} * mip->height * (bc4 ? 1 : 4);
                if (linear_size > SnapshotLimit - snapshot_bytes ||
                    decoded_size > SnapshotLimit - snapshot_bytes - linear_size)
                    return reject("native compressed decode exceeds snapshot capacity");
                snapshot_bytes += linear_size + decoded_size;
                std::vector<u8> linear(static_cast<size_t>(linear_size));
                Tegra::Texture::UnswizzleTexture(linear, tiled, bytes_per_block,
                                                 static_cast<u32>(columns), static_cast<u32>(rows),
                                                 1, mip->block_height, 0, 1,
                                                 Tegra::Texture::BlockLinearLayout::TuringColor2D);
                uploaded_pixels.resize(static_cast<size_t>(decoded_size));
                if (bc4) {
                    // The decoder emits R8. Its block-row stride uses a full
                    // four-pixel block except when the entire image is narrower.
                    VideoCommon::BufferImageCopy copy{
                        .buffer_row_length = static_cast<u32>(columns) * (std::min)(mip->width, 4U),
                        .image_subresource = {},
                        .image_extent = sampled.size,
                    };
                    VideoCommon::DecompressBCn(linear, uploaded_pixels, copy, sampled.format);
                } else {
                    Tegra::Texture::ASTC::Decompress(linear, mip->width, mip->height, 1, 4, 4,
                                                     uploaded_pixels);
                }
                if (stop_token.stop_requested())
                    return false;
            }
            // This standalone image represents the selected guest mip, so its
            // host LOD is zero even when the guest's fixed clamp selects LOD > 0.
            native_samplers.push_back(device.GetLogical().CreateSampler(VkSamplerCreateInfo{
                .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .magFilter = sampler.mag_filter == 1 ? VK_FILTER_NEAREST : VK_FILTER_LINEAR,
                .minFilter = sampler.min_filter == 1 ? VK_FILTER_NEAREST : VK_FILTER_LINEAR,
                .mipmapMode = sampler.mip_filter == 3 ? VK_SAMPLER_MIPMAP_MODE_LINEAR
                                                      : VK_SAMPLER_MIPMAP_MODE_NEAREST,
                .addressModeU = sampler.wrap[0] == 0 ? VK_SAMPLER_ADDRESS_MODE_REPEAT
                                                     : VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                .addressModeV = sampler.wrap[1] == 0 ? VK_SAMPLER_ADDRESS_MODE_REPEAT
                                                     : VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                .addressModeW = sampler.wrap[2] == 0 ? VK_SAMPLER_ADDRESS_MODE_REPEAT
                                                     : VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                .mipLodBias = 0.f,
                .anisotropyEnable = VK_FALSE,
                .maxAnisotropy = 1.f,
                .compareEnable = VK_FALSE,
                .compareOp = VK_COMPARE_OP_ALWAYS,
                .minLod = 0.f,
                .maxLod = 0.f,
                .borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
                .unnormalizedCoordinates = VK_FALSE,
            }));
            sampled_requests.push_back({stage, i, mip->address, *backing, sampled, sampled_format,
                                        *native_samplers.back(), std::move(uploaded_pixels)});
            LOG_DEBUG(HW_GPU,
                      "Native sampled mip: level={}, address={:#x}, extent={}x{}x{}, "
                      "block_height={}, block_depth={}",
                      mip->level, mip->address, mip->width, mip->height, mip->depth,
                      mip->block_height, mip->block_depth);
        }
    }
    {
        std::scoped_lock lock{buffer_cache.mutex};
        for (u32 index = 0; index < targets->count; ++index)
            if (buffer_cache.IsRegionGpuModified(outputs[index],
                                                 targets->targets[index].color.size_bytes))
                return reject("native target overlaps pending buffer writes");
    }
    // Keep pixels private. Include the first volume samples after earlier
    // initialization draws have exhausted the ordinary readback budget.
    const bool has_volume = std::ranges::any_of(sampled_requests, [](const auto& sampled) {
        return sampled.info.type == VideoCommon::ImageType::e3D;
    });
    const bool record_readbacks =
        native_draw_readbacks < 32 || (targets->count > 1 && native_mrt_readbacks < 2) ||
        (has_volume && native_volume_readbacks < 2) ||
        (target->format == Clear::Format::RGBA8Unorm && native_unorm_readbacks < 2);
    std::array<std::vector<u8>, 8> readbacks;
    std::array<u64, 8> before_hashes{};
    if (record_readbacks) {
        for (u32 index = 0; index < targets->count; ++index) {
            const auto& color = targets->targets[index].color;
            auto& readback = readbacks[index];
            readback.resize(color.size_bytes);
            if (!lease.ReadBlockChecked(color.address, readback))
                return reject("native target initial snapshot");
            before_hashes[index] =
                Common::CityHash64(reinterpret_cast<const char*>(readback.data()), readback.size());
        }
    }
    std::array<TextureCache::NativeColorAttachment, 8> attachments;
    for (u32 index = 0; index < targets->count; ++index) {
        const auto& color = targets->targets[index].color;
        VideoCommon::ImageInfo image;
        image.format = format;
        image.type = VideoCommon::ImageType::e2D;
        image.size = {color.width, color.height, 1};
        image.block = {0, color.block_height, 0};
        image.layer_stride = image.maybe_unaligned_layer_stride = color.size_bytes;
        image.layout = Tegra::Texture::BlockLinearLayout::TuringColor2D;
        if (VideoCommon::CalculateGuestSizeInBytes(image) != color.size_bytes)
            return reject("native target footprint mismatch");
        attachments[index] = {image, color.address, outputs[index]};
    }
    const auto record = [&] {
        std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
        std::vector<GraphicsPipeline::NativeSampledImage> sampled_images;
        for (const auto& sampled : sampled_requests) {
            if (stop_token.stop_requested())
                return false;
            if (sampled.info.type == VideoCommon::ImageType::e3D) {
                const u64 size = VideoCommon::CalculateGuestSizeInBytes(sampled.info);
                if (size > SnapshotLimit - native_sample_copy_bytes)
                    return false;
                native_sample_copy_bytes += size;
                // Assemble a private logical volume from independently known
                // one-GOB slices. Do not register an overlapping guest volume.
                native_sample_images.push_back(
                    std::make_unique<Image>(texture_cache_runtime, sampled.info, 0, 0));
                auto& volume_image = *native_sample_images.back();
                texture_cache_runtime.TransitionImageLayout(volume_image);
                auto slice = sampled.info;
                slice.type = VideoCommon::ImageType::e2D;
                slice.size.depth = 1;
                slice.block.depth = 0;
                slice.layer_stride = slice.maybe_unaligned_layer_stride = 512;
                if (VideoCommon::CalculateGuestSizeInBytes(slice) != 512)
                    return false;
                for (u32 z = 0; z < sampled.info.size.depth; ++z) {
                    if (stop_token.stop_requested())
                        return false;
                    const u64 offset = u64{z} * 512;
                    const auto source = texture_cache.PrepareNativeClearTarget(
                        slice, sampled.address + offset, sampled.backing + offset, true, true);
                    if (!source || buffer_cache.IsRegionGpuModified(sampled.backing + offset, 512))
                        return false;
                    const VideoCommon::ImageCopy copy{
                        .src_subresource = {},
                        .dst_subresource = {},
                        .src_offset = {},
                        .dst_offset = {0, 0, static_cast<s32>(z)},
                        .extent = slice.size,
                    };
                    texture_cache_runtime.CopyImage(volume_image, *source->image,
                                                    std::span{&copy, 1});
                }
                native_sample_views.push_back(
                    device.GetLogical().CreateImageView(VkImageViewCreateInfo{
                        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                        .pNext = nullptr,
                        .flags = 0,
                        .image = volume_image.Handle(),
                        .viewType = VK_IMAGE_VIEW_TYPE_3D,
                        .format = sampled.host_format,
                        .components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                                       VK_COMPONENT_SWIZZLE_IDENTITY,
                                       VK_COMPONENT_SWIZZLE_IDENTITY},
                        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
                    }));
                sampled_images.push_back(
                    {sampled.stage, sampled.index, *native_sample_views.back(), sampled.sampler});
                LOG_DEBUG(HW_GPU, "Native sampled color volume: address={:#x}, extent={}x{}x{}",
                          sampled.address, sampled.info.size.width, sampled.info.size.height,
                          sampled.info.size.depth);
                continue;
            }
            if (!sampled.uploaded_pixels.empty()) {
                // Keep the decoded asset private to this draw. It must not be
                // registered as an overlapping interpretation of guest blocks.
                if (buffer_cache.IsRegionGpuModified(
                        sampled.backing, VideoCommon::CalculateGuestSizeInBytes(sampled.info)) ||
                    texture_cache.IsRegionGpuModified(
                        sampled.backing, VideoCommon::CalculateGuestSizeInBytes(sampled.info)))
                    return false;
                auto host_info = sampled.info;
                const bool bc4 = sampled.info.format == VideoCore::Surface::PixelFormat::BC4_UNORM;
                host_info.format = bc4 ? VideoCore::Surface::PixelFormat::R8_UNORM
                                       : VideoCore::Surface::PixelFormat::A8B8G8R8_SRGB;
                const u64 host_size = VideoCommon::CalculateGuestSizeInBytes(host_info);
                if (host_size > SnapshotLimit - native_sample_copy_bytes)
                    return false;
                native_sample_copy_bytes += host_size;
                host_info.layer_stride = host_info.maybe_unaligned_layer_stride =
                    static_cast<u32>(host_size);
                native_sample_images.push_back(
                    std::make_unique<Image>(texture_cache_runtime, host_info, 0, 0));
                auto& uploaded = *native_sample_images.back();
                const auto staging =
                    staging_pool.Request(sampled.uploaded_pixels.size(), MemoryUsage::Upload);
                if (!staging.buffer || staging.mapped_span.size() < sampled.uploaded_pixels.size())
                    return false;
                std::memcpy(staging.mapped_span.data(), sampled.uploaded_pixels.data(),
                            sampled.uploaded_pixels.size());
                const VideoCommon::BufferImageCopy copy{
                    .buffer_offset = 0,
                    .buffer_size = sampled.uploaded_pixels.size(),
                    .buffer_row_length = sampled.info.size.width,
                    .buffer_image_height = sampled.info.size.height,
                    .image_subresource = {},
                    .image_offset = {},
                    .image_extent = sampled.info.size,
                };
                uploaded.UploadMemory(staging, std::span{&copy, 1});
                native_sample_views.push_back(
                    device.GetLogical().CreateImageView(VkImageViewCreateInfo{
                        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                        .pNext = nullptr,
                        .flags = 0,
                        .image = uploaded.Handle(),
                        .viewType = VK_IMAGE_VIEW_TYPE_2D,
                        .format = sampled.host_format,
                        .components =
                            bc4 ? VkComponentMapping{VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R,
                                                     VK_COMPONENT_SWIZZLE_R,
                                                     VK_COMPONENT_SWIZZLE_ONE}
                                : VkComponentMapping{VK_COMPONENT_SWIZZLE_IDENTITY,
                                                     VK_COMPONENT_SWIZZLE_IDENTITY,
                                                     VK_COMPONENT_SWIZZLE_IDENTITY,
                                                     VK_COMPONENT_SWIZZLE_IDENTITY},
                        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
                    }));
                sampled_images.push_back(
                    {sampled.stage, sampled.index, *native_sample_views.back(), sampled.sampler});
                LOG_DEBUG(
                    HW_GPU,
                    "Native compressed upload: format={}, address={:#x}, extent={}x{}, bytes={}",
                    sampled.info.format, sampled.address, sampled.info.size.width,
                    sampled.info.size.height, sampled.uploaded_pixels.size());
                continue;
            }
            // The descriptor's opaque registration preset cannot identify a
            // layout. Only a previously established exact native image may be sampled.
            const auto source = texture_cache.PrepareNativeClearTarget(
                sampled.info, sampled.address, sampled.backing, true, true);
            if (!source ||
                buffer_cache.IsRegionGpuModified(
                    sampled.backing, VideoCommon::CalculateGuestSizeInBytes(sampled.info)))
                return false;
            VkImageView view =
                texture_cache.GetImageView(source->view_id).Handle(Shader::TextureType::Color2D);
            if (source->image->info.size.width != sampled.info.size.width) {
                // Equal guest pitch proves the same pixel rows, but normalized
                // coordinates and clamp-to-edge need the logical texture width.
                // Copy those pixels to a standalone host image without registering
                // another overlapping guest image in the cache.
                const u64 size = VideoCommon::CalculateGuestSizeInBytes(sampled.info);
                if (size > SnapshotLimit - native_sample_copy_bytes)
                    return false;
                native_sample_copy_bytes += size;
                native_sample_images.push_back(
                    std::make_unique<Image>(texture_cache_runtime, sampled.info, 0, 0));
                auto& copy_image = *native_sample_images.back();
                texture_cache_runtime.TransitionImageLayout(copy_image);
                const VideoCommon::ImageCopy copy{
                    .src_subresource = {},
                    .dst_subresource = {},
                    .src_offset = {},
                    .dst_offset = {},
                    .extent = sampled.info.size,
                };
                texture_cache_runtime.CopyImage(copy_image, *source->image, std::span{&copy, 1});
                native_sample_views.push_back(
                    device.GetLogical().CreateImageView(VkImageViewCreateInfo{
                        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                        .pNext = nullptr,
                        .flags = 0,
                        .image = copy_image.Handle(),
                        .viewType = VK_IMAGE_VIEW_TYPE_2D,
                        .format = sampled.host_format,
                        .components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                                       VK_COMPONENT_SWIZZLE_IDENTITY,
                                       VK_COMPONENT_SWIZZLE_IDENTITY},
                        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
                    }));
                view = *native_sample_views.back();
                LOG_DEBUG(
                    HW_GPU,
                    "Native sampled color crop: storage_width={}, logical_width={}, height={}",
                    source->image->info.size.width, sampled.info.size.width,
                    sampled.info.size.height);
            }
            if (!view)
                return false;
            sampled_images.push_back({sampled.stage, sampled.index, view, sampled.sampler});
        }
        const auto framebuffer =
            texture_cache.PrepareNativeColorTargets(std::span{attachments}.first(targets->count));
        if (!sampled_images.empty()) {
            scheduler.RequestOutsideRenderPassOperationContext();
            scheduler.Record([](vk::CommandBuffer command) {
                const VkMemoryBarrier barrier{
                    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
                    .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
                };
                command.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                        VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                                            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                        0, barrier);
            });
        }
        if (!framebuffer || stop_token.stop_requested() ||
            framebuffer->framebuffer->NumColorBuffers() != targets->count ||
            framebuffer->framebuffer->RenderArea().width != target->width ||
            framebuffer->framebuffer->RenderArea().height != target->height ||
            !pipeline->ConfigureNative(uniforms, sampled_images, framebuffer->framebuffer,
                                       {static_cast<float>(regs.surface_clip.width),
                                        static_cast<float>(regs.surface_clip.height)}))
            return false;
        // Native predicates are resolved explicitly by the engine. Keep any
        // legacy conditional-rendering state out of this checked draw.
        query_cache.NotifySegment(false);
        query_cache.CounterEnable(VideoCommon::QueryType::ZPassPixelCount64, read(0x545) != 0);
        const auto index = staging_pool.Request(indices.size(), MemoryUsage::Upload);
        if (!index.buffer || index.mapped_span.size() < indices.size() ||
            index.offset % (index_format == 2 ? 4 : 2))
            return false;
        std::memcpy(index.mapped_span.data(), indices.data(), indices.size());
        for (u32 stream = 0; stream < vertices.size(); ++stream) {
            if (stop_token.stop_requested())
                return false;
            if (vertices[stream].empty())
                continue;
            const auto buffer = staging_pool.Request(vertices[stream].size(), MemoryUsage::Upload);
            if (!buffer.buffer || buffer.mapped_span.size() < vertices[stream].size() ||
                buffer.offset % 4)
                return false;
            std::memcpy(buffer.mapped_span.data(), vertices[stream].data(),
                        vertices[stream].size());
            scheduler.Record([stream, handle = buffer.buffer,
                              offset = buffer.offset](vk::CommandBuffer command) {
                command.BindVertexBuffers(stream, 1, &handle, &offset);
            });
        }
        if (stop_token.stop_requested())
            return false;
        const u32 viewport_count = (std::min)(device.GetMaxViewports(), u32{Maxwell::NumViewports});
        scheduler.Record(
            [viewport, scissor, viewport_count, index_buffer = index.buffer,
             index_offset = index.offset, host_index_format, count = request.count, stop_token,
             blend_constants,
             depth_bounds = device.IsDepthBoundsSupported()](vk::CommandBuffer command) {
                if (stop_token.stop_requested())
                    return;
                std::array<VkViewport, Maxwell::NumViewports> viewports;
                std::array<VkRect2D, Maxwell::NumViewports> scissors;
                viewports.fill(viewport);
                scissors.fill(scissor);
                command.SetViewport(0, {viewports.data(), viewport_count});
                command.SetScissor(0, {scissors.data(), viewport_count});
                command.SetDepthBias(0.f, 0.f, 0.f);
                command.SetBlendConstants(blend_constants.data());
                if (depth_bounds)
                    command.SetDepthBounds(0.f, 1.f);
                command.SetStencilCompareMask(VK_STENCIL_FACE_FRONT_AND_BACK, 0xffffffff);
                command.SetStencilWriteMask(VK_STENCIL_FACE_FRONT_AND_BACK, 0xffffffff);
                command.SetStencilReference(VK_STENCIL_FACE_FRONT_AND_BACK, 0);
                command.SetLineWidth(1.f);
                command.BindIndexBuffer(index_buffer, index_offset, host_index_format);
                command.DrawIndexed(count, 1, 0, 0, 0);
            });
        for (u32 attachment_index = 0; attachment_index < framebuffer->count; ++attachment_index)
            texture_cache.MarkModification(framebuffer->image_ids[attachment_index]);
        return true;
    };
    const bool recorded = record();
    scheduler.Finish();
    if (!recorded) {
        work_drained = true;
        return reject("native graphics resource binding");
    }
    {
        std::scoped_lock lock{texture_cache.mutex};
        for (u32 index = 0; index < targets->count; ++index)
            texture_cache.DownloadMemory(outputs[index], targets->targets[index].color.size_bytes);
    }
    {
        std::scoped_lock lock{buffer_cache.mutex};
        for (u32 index = 0; index < targets->count; ++index)
            buffer_cache.WriteMemory(outputs[index], targets->targets[index].color.size_bytes);
    }
    if (stop_token.stop_requested())
        return false;
    if (record_readbacks) {
        for (u32 index = 0; index < targets->count; ++index) {
            auto& readback = readbacks[index];
            if (!lease.ReadBlockChecked(targets->targets[index].color.address, readback))
                return reject("native target final readback");
            const u64 after_hash =
                Common::CityHash64(reinterpret_cast<const char*>(readback.data()), readback.size());
            LOG_DEBUG(HW_GPU,
                      "Native draw readback: target={}, before={:#x}, after={:#x}, changed={}",
                      index, before_hashes[index], after_hash, before_hashes[index] != after_hash);
        }
        ++native_draw_readbacks;
        if (targets->count > 1)
            ++native_mrt_readbacks;
        if (has_volume)
            ++native_volume_readbacks;
        if (target->format == Clear::Format::RGBA8Unorm)
            ++native_unorm_readbacks;
    }
    work_drained = true;
    LOG_DEBUG(
        HW_GPU,
        "Native indexed draw completed: count={}, minimum={}, maximum={}, extent={}x{}, targets={}",
        request.count, bounds->minimum, bounds->maximum, target->width, target->height,
        targets->count);
    return true;
}

bool RasterizerVulkan::InvalidateNativeCaches(
    const Tegra::Control::NativeCacheInvalidation& action) {
    scheduler.Finish();
    gpu.InvalidateGPUCache();
    if (action.instructions) {
        pipeline_cache.SyncGuestHost();
    }
    // Native code/CBUF bytes are freshly checked before every dispatch. SSBOs stay
    // GPU-resident and the dispatch barrier orders prior writes; never discard their
    // GPU-modified contents as if this were a CPU overwrite. Native descriptors
    // are captured separately; invalidate the legacy descriptor tables here.
    if (action.texture_headers || action.samplers) {
        std::scoped_lock lock{texture_cache.mutex};
        return texture_cache.InvalidateDescriptorTables(action.texture_headers, action.samplers);
    }
    return true;
}

void RasterizerVulkan::Query(GPUVAddr gpu_addr, VideoCommon::QueryType type,
                             VideoCommon::QueryPropertiesFlags flags, u32 payload, u32 subreport) {
    query_cache.CounterReport(gpu_addr, type, flags, payload, subreport);
}

void RasterizerVulkan::BindGraphicsUniformBuffer(size_t stage, u32 index, GPUVAddr gpu_addr,
                                                 u32 size) {
    buffer_cache.BindGraphicsUniformBuffer(stage, index, gpu_addr, size);
}

void Vulkan::RasterizerVulkan::DisableGraphicsUniformBuffer(size_t stage, u32 index) {
    buffer_cache.DisableGraphicsUniformBuffer(stage, index);
}

void RasterizerVulkan::FlushAll() {}

void RasterizerVulkan::FlushRegion(DAddr addr, u64 size, VideoCommon::CacheType which) {
    if (addr == 0 || size == 0) {
        return;
    }
    if (True(which & VideoCommon::CacheType::TextureCache)) {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.DownloadMemory(addr, size);
    }
    if ((True(which & VideoCommon::CacheType::BufferCache))) {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.DownloadMemory(addr, size);
    }
    if ((True(which & VideoCommon::CacheType::QueryCache))) {
        query_cache.FlushRegion(addr, size);
    }
}

bool RasterizerVulkan::MustFlushRegion(DAddr addr, u64 size, VideoCommon::CacheType which) {
    if ((True(which & VideoCommon::CacheType::BufferCache))) {
        std::scoped_lock lock{buffer_cache.mutex};
        if (buffer_cache.IsRegionGpuModified(addr, size)) {
            return true;
        }
    }
    if (!Settings::IsGPULevelHigh()) {
        return false;
    }
    if (True(which & VideoCommon::CacheType::TextureCache)) {
        std::scoped_lock lock{texture_cache.mutex};
        return texture_cache.IsRegionGpuModified(addr, size);
    }
    return false;
}

VideoCore::RasterizerDownloadArea RasterizerVulkan::GetFlushArea(DAddr addr, u64 size) {
    {
        std::scoped_lock lock{texture_cache.mutex};
        auto area = texture_cache.GetFlushArea(addr, size);
        if (area) {
            return *area;
        }
    }
    VideoCore::RasterizerDownloadArea new_area{
        .start_address = Common::AlignDown(addr, Core::DEVICE_PAGESIZE),
        .end_address = Common::AlignUp(addr + size, Core::DEVICE_PAGESIZE),
        .preemtive = true,
    };
    return new_area;
}

void RasterizerVulkan::InvalidateRegion(DAddr addr, u64 size, VideoCommon::CacheType which) {
    if (addr == 0 || size == 0) {
        return;
    }
    if (True(which & VideoCommon::CacheType::TextureCache)) {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.WriteMemory(addr, size);
    }
    if ((True(which & VideoCommon::CacheType::BufferCache))) {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.WriteMemory(addr, size);
    }
    if ((True(which & VideoCommon::CacheType::QueryCache))) {
        query_cache.InvalidateRegion(addr, size);
    }
    if ((True(which & VideoCommon::CacheType::ShaderCache))) {
        pipeline_cache.InvalidateRegion(addr, size);
    }
}

void RasterizerVulkan::InnerInvalidation(std::span<const std::pair<DAddr, std::size_t>> sequences) {
    {
        std::scoped_lock lock{texture_cache.mutex};
        for (const auto& [addr, size] : sequences) {
            texture_cache.WriteMemory(addr, size);
        }
    }
    {
        std::scoped_lock lock{buffer_cache.mutex};
        for (const auto& [addr, size] : sequences) {
            buffer_cache.WriteMemory(addr, size);
        }
    }
    {
        for (const auto& [addr, size] : sequences) {
            query_cache.InvalidateRegion(addr, size);
            pipeline_cache.InvalidateRegion(addr, size);
        }
    }
}

bool RasterizerVulkan::OnCPUWrite(DAddr addr, u64 size) {
    DEBUG_ASSERT(addr != 0 || size != 0);
    {
        std::scoped_lock lock{buffer_cache.mutex};
        if (buffer_cache.OnCPUWrite(addr, size)) {
            return true;
        }
    }
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.WriteMemory(addr, size);
    }
    pipeline_cache.InvalidateRegion(addr, size);
    return false;
}

void RasterizerVulkan::OnCacheInvalidation(DAddr addr, u64 size) {
    if (addr == 0 || size == 0) {
        return;
    }

    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.WriteMemory(addr, size);
    }
    {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.WriteMemory(addr, size);
    }
    pipeline_cache.InvalidateRegion(addr, size);
}

bool RasterizerVulkan::TryCacheInvalidation(DAddr addr, u64 size) {
    if (addr == 0 || size == 0) return true;
    {
        std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
        if (buffer_cache.IsRegionGpuModified(addr, size)) return false;
        texture_cache.WriteMemory(addr, size);
        buffer_cache.WriteMemory(addr, size);
    }
    pipeline_cache.InvalidateRegion(addr, size);
    return true;
}

void RasterizerVulkan::InvalidateGPUCache() {
    gpu.InvalidateGPUCache();
}

void RasterizerVulkan::UnmapMemory(DAddr addr, u64 size) {
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.UnmapMemory(addr, size);
    }
    {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.WriteMemory(addr, size);
    }
    pipeline_cache.OnCacheInvalidation(addr, size);
}

void RasterizerVulkan::ModifyGPUMemory(size_t as_id, GPUVAddr addr, u64 size) {
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.UnmapGPUMemory(as_id, addr, size);
    }
    buffer_cache.UnmapGPUMemory(as_id, addr, size);
}

void RasterizerVulkan::SignalFence(std::function<void()>&& func) {
    fence_manager.SignalFence(std::move(func));
}

void RasterizerVulkan::SyncOperation(std::function<void()>&& func) {
    fence_manager.SyncOperation(std::move(func));
}

void RasterizerVulkan::SignalSyncPoint(u32 value) {
    fence_manager.SignalSyncPoint(value);
}

void RasterizerVulkan::SignalReference() {
    fence_manager.SignalReference();
}

void RasterizerVulkan::ReleaseFences(bool force) {
    fence_manager.WaitPendingFences(force);
}

void RasterizerVulkan::FlushAndInvalidateRegion(DAddr addr, u64 size,
                                                VideoCommon::CacheType which) {
    if (Settings::IsGPULevelHigh()) {
        FlushRegion(addr, size, which);
    }
    InvalidateRegion(addr, size, which);
}

void RasterizerVulkan::WaitForIdle() {
    // Everything but wait pixel operations. This intentionally includes FRAGMENT_SHADER_BIT because
    // fragment shaders can still write storage buffers.
    VkPipelineStageFlags flags =
        VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_VERTEX_INPUT_BIT |
        VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT |
        VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT |
        VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    if (device.IsExtTransformFeedbackSupported()) {
        flags |= VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT;
    }

    query_cache.NotifyWFI();

    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([event = *wfi_event, flags](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetEvent(event, flags);
        cmdbuf.WaitEvents(event, flags, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, {}, {}, {});
    });
    fence_manager.SignalOrdering();
}

void RasterizerVulkan::FragmentBarrier() {
    // We already put barriers when a render pass finishes
    scheduler.RequestOutsideRenderPassOperationContext();
}

void RasterizerVulkan::TiledCacheBarrier() {
    // TODO: Implementing tiled barriers requires rewriting a good chunk of the Vulkan backend
}

void RasterizerVulkan::FlushCommands() {
    if (draw_counter == 0) {
        return;
    }
    draw_counter = 0;
    scheduler.Flush();
}

void RasterizerVulkan::TickFrame() {
    draw_counter = 0;
    guest_descriptor_queue.TickFrame();
    compute_pass_descriptor_queue.TickFrame();
    descriptor_buffer_ring.TickFrame();
    fence_manager.TickFrame();
    staging_pool.TickFrame();
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.TickFrame();
    }
    {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.TickFrame();
    }
}

bool RasterizerVulkan::AccelerateConditionalRendering() {
    gpu_memory->FlushCaching();
    return query_cache.AccelerateHostConditionalRendering();
}

bool RasterizerVulkan::HasDrawTransformFeedback() {
    return device.IsTransformFeedbackDrawSupported();
}

bool RasterizerVulkan::AccelerateSurfaceCopy(const Tegra::Engines::Fermi2D::Surface& src,
                                             const Tegra::Engines::Fermi2D::Surface& dst,
                                             const Tegra::Engines::Fermi2D::Config& copy_config) {
    std::scoped_lock lock{texture_cache.mutex};
    return texture_cache.BlitImage(dst, src, copy_config);
}

Tegra::Engines::AccelerateDMAInterface& RasterizerVulkan::AccessAccelerateDMA() {
    return accelerate_dma;
}

void RasterizerVulkan::AccelerateInlineToMemory(GPUVAddr address, size_t copy_size,
                                                std::span<const u8> memory) {
    auto cpu_addr = gpu_memory->GpuToCpuAddress(address);
    if (!cpu_addr) [[unlikely]] {
        gpu_memory->WriteBlock(address, memory.data(), copy_size);
        return;
    }
    gpu_memory->WriteBlockUnsafe(address, memory.data(), copy_size);
    {
        std::unique_lock<std::recursive_mutex> lock{buffer_cache.mutex};
        if (!buffer_cache.InlineMemory(*cpu_addr, copy_size, memory)) {
            buffer_cache.WriteMemory(*cpu_addr, copy_size);
        }
    }
    {
        std::scoped_lock lock_texture{texture_cache.mutex};
        texture_cache.WriteMemory(*cpu_addr, copy_size);
    }
    pipeline_cache.InvalidateRegion(*cpu_addr, copy_size);
    query_cache.InvalidateRegion(*cpu_addr, copy_size);
}

std::optional<FramebufferTextureInfo> RasterizerVulkan::AccelerateDisplay(
    const Tegra::FramebufferConfig& config, DAddr framebuffer_addr, u32 pixel_stride) {
    if (!framebuffer_addr) {
        return {};
    }
    std::scoped_lock lock{texture_cache.mutex};
    const auto [image_view, scaled] =
        texture_cache.TryFindFramebufferImageView(config, framebuffer_addr);
    if (!image_view) {
        return {};
    }

    query_cache.NotifySegment(false);

    const auto& resolution = Settings::values.resolution_info;

    FramebufferTextureInfo info{};
    info.image = image_view->ImageHandle();
    info.image_view = image_view->Handle(Shader::TextureType::Color2D);
    info.width = image_view->size.width;
    info.height = image_view->size.height;
    info.scaled_width = scaled ? resolution.ScaleUp(info.width) : info.width;
    info.scaled_height = scaled ? resolution.ScaleUp(info.height) : info.height;
    return info;
}

void RasterizerVulkan::LoadDiskResources(u64 title_id, std::stop_token stop_loading,
                                         const VideoCore::DiskResourceLoadCallback& callback) {
    pipeline_cache.LoadDiskResources(title_id, stop_loading, callback);
}

void RasterizerVulkan::FlushWork() {
#ifdef __ANDROID__
    static constexpr u32 DRAWS_TO_DISPATCH = 512;
    static constexpr u32 CHECK_MASK = 3;
#else
    static constexpr u32 DRAWS_TO_DISPATCH = 4096;
    static constexpr u32 CHECK_MASK = 7;
#endif // __ANDROID__

    static_assert(DRAWS_TO_DISPATCH % (CHECK_MASK + 1) == 0);
    if ((++draw_counter & CHECK_MASK) != CHECK_MASK) {
        return;
    }
    if (draw_counter < DRAWS_TO_DISPATCH) {
        scheduler.DispatchWork();
        return;
    }
    scheduler.Flush();
    draw_counter = 0;
}

AccelerateDMA::AccelerateDMA(BufferCache& buffer_cache_, TextureCache& texture_cache_,
                             Scheduler& scheduler_)
    : buffer_cache{buffer_cache_}, texture_cache{texture_cache_}, scheduler{scheduler_} {}

bool AccelerateDMA::BufferClear(GPUVAddr src_address, u64 amount, u32 value) {
    std::scoped_lock lock{buffer_cache.mutex};
    return buffer_cache.DMAClear(src_address, amount, value);
}

bool AccelerateDMA::BufferCopy(GPUVAddr src_address, GPUVAddr dest_address, u64 amount) {
    std::scoped_lock lock{buffer_cache.mutex};
    return buffer_cache.DMACopy(src_address, dest_address, amount);
}

template <bool IS_IMAGE_UPLOAD>
bool AccelerateDMA::DmaBufferImageCopy(const Tegra::DMA::ImageCopy& copy_info,
                                       const Tegra::DMA::BufferOperand& buffer_operand,
                                       const Tegra::DMA::ImageOperand& image_operand) {
    std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
    const auto image_id = texture_cache.DmaImageId(image_operand, IS_IMAGE_UPLOAD);
    if (image_id == VideoCommon::NULL_IMAGE_ID) {
        return false;
    }
    const u32 buffer_size = static_cast<u32>(buffer_operand.pitch * buffer_operand.height);
    static constexpr auto sync_info = VideoCommon::ObtainBufferSynchronize::FullSynchronize;
    const auto post_op = IS_IMAGE_UPLOAD ? VideoCommon::ObtainBufferOperation::DoNothing
                                         : VideoCommon::ObtainBufferOperation::MarkAsWritten;
    const auto [buffer, offset] =
        buffer_cache.ObtainBuffer(buffer_operand.address, buffer_size, sync_info, post_op);

    const auto [image, copy] = texture_cache.DmaBufferImageCopy(
        copy_info, buffer_operand, image_operand, image_id, IS_IMAGE_UPLOAD);
    const std::span copy_span{&copy, 1};

    if constexpr (IS_IMAGE_UPLOAD) {
        texture_cache.PrepareImage(image_id, true, false);
        image->UploadMemory(buffer->Handle(), offset, copy_span);
    } else {
        if (offset % BytesPerBlock(image->info.format)) {
            return false;
        }
        texture_cache.DownloadImageIntoBuffer(image, buffer->Handle(), offset, copy_span,
                                              buffer_operand.address, buffer_size);
    }
    return true;
}

bool AccelerateDMA::ImageToBuffer(const Tegra::DMA::ImageCopy& copy_info,
                                  const Tegra::DMA::ImageOperand& image_operand,
                                  const Tegra::DMA::BufferOperand& buffer_operand) {
    return DmaBufferImageCopy<false>(copy_info, buffer_operand, image_operand);
}

bool AccelerateDMA::BufferToImage(const Tegra::DMA::ImageCopy& copy_info,
                                  const Tegra::DMA::BufferOperand& buffer_operand,
                                  const Tegra::DMA::ImageOperand& image_operand) {
    return DmaBufferImageCopy<true>(copy_info, buffer_operand, image_operand);
}

void RasterizerVulkan::UpdateDynamicStates() {
    auto& regs = maxwell3d->regs;
    auto& flags = maxwell3d->dirty.flags;
    const auto topology = maxwell3d->draw_manager.draw_state.topology;
    const bool topology_changed = state_tracker.ChangePrimitiveTopology(topology);
    if (topology_changed) {
        flags[Dirty::DepthBiasEnable] = true;
        flags[Dirty::PrimitiveRestartEnable] = true;
    }

    UpdateViewportsState(regs);
    UpdateScissorsState(regs);
    UpdateDepthBias(regs);
    UpdateBlendConstants(regs);
    UpdateDepthBounds(regs);
    UpdateStencilFaces(regs);
    UpdateLineWidth(regs);

    if (device.IsExtExtendedDynamicStateSupported()) {
        UpdateCullMode(regs);
        UpdateDepthCompareOp(regs);
        UpdateFrontFace(regs);
        UpdateStencilOp(regs);
        if (state_tracker.TouchStateEnable()) {
            UpdateDepthBoundsTestEnable(regs);
            UpdateDepthTestEnable(regs);
            UpdateDepthWriteEnable(regs);
            UpdateStencilTestEnable(regs);
        }
        if (topology_changed) {
            scheduler.Record([topology_vk = MaxwellToVK::PrimitiveTopology(device, topology)](
                                 vk::CommandBuffer cmdbuf) {
                cmdbuf.SetPrimitiveTopologyEXT(topology_vk);
            });
        }
    }

    if (device.IsExtExtendedDynamicState2Supported()) {
        UpdatePrimitiveRestartEnable(regs);
        UpdateRasterizerDiscardEnable(regs);
        UpdateDepthBiasEnable(regs);
    }

    if (device.IsExtExtendedDynamicState2ExtrasSupported()) {
        UpdateLogicOp(regs);
    }

    if (device.IsExtExtendedDynamicState3EnablesSupported()) {
        using namespace Tegra::Engines;
        // AMD Workaround: LogicOp incompatible with float render targets
        if (device.GetDriverID() == VkDriverIdKHR::VK_DRIVER_ID_AMD_OPEN_SOURCE ||
            device.GetDriverID() == VkDriverIdKHR::VK_DRIVER_ID_AMD_PROPRIETARY) {
            const auto has_float = std::any_of(
                regs.vertex_attrib_format.begin(), regs.vertex_attrib_format.end(),
                [](const auto& attrib) {
                    return attrib.type == Maxwell3D::Regs::VertexAttribute::Type::Float;
                }
            );
            if (regs.logic_op.enable) {
                regs.logic_op.enable = static_cast<u32>(!has_float);
            }
        }
        UpdateLogicOpEnable(regs);
        UpdateDepthClampEnable(regs);
        UpdateLineRasterizationMode(regs);
        UpdateLineStippleEnable(regs);
        UpdateConservativeRasterizationMode(regs);
        UpdateAlphaToCoverageEnable(regs);
        UpdateAlphaToOneEnable(regs);
    }

    if (device.IsExtExtendedDynamicState3BlendingSupported()) {
        UpdateBlending(regs);
    } else if (device.IsExtColorWriteEnableSupported()) {
        UpdateColorWriteEnable(regs);
    }

    if (device.IsExtVertexInputDynamicStateSupported()) {
        if (auto* gp = pipeline_cache.CurrentGraphicsPipeline(); gp && gp->HasDynamicVertexInput()) {
            UpdateVertexInput(regs);
        }
    }
}

void RasterizerVulkan::HandleTransformFeedback() {
    static std::once_flag warn_unsupported;

    const auto& regs = maxwell3d->regs;
    if (!device.IsExtTransformFeedbackSupported()) {
        if (regs.transform_feedback_enabled != 0) {
            std::call_once(warn_unsupported, [&] {
                LOG_WARNING(Render_Vulkan, "Transform feedback requested by guest but VK_EXT_transform_feedback is unavailable; queries disabled");
            });
        } else {
            std::call_once(warn_unsupported, [&] {
                LOG_INFO(Render_Vulkan, "VK_EXT_transform_feedback not available on device");
            });
        }
        return;
    }
    query_cache.CounterEnable(VideoCommon::QueryType::StreamingByteCount,
                              regs.transform_feedback_enabled);
    if (regs.transform_feedback_enabled != 0) {
        // Log extension usage for transform feedback
        if (GPU::Logging::IsActive()) {
            GPU::Logging::GPULogger::GetInstance().LogExtensionUsage(
                "VK_EXT_transform_feedback", "HandleTransformFeedback");
        }
        UNIMPLEMENTED_IF(regs.IsShaderConfigEnabled(Maxwell::ShaderType::TessellationInit) ||
                         regs.IsShaderConfigEnabled(Maxwell::ShaderType::Tessellation));
    }
}

void RasterizerVulkan::UpdateViewportsState(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchViewports()) {
        return;
    }

    maxwell3d->dirty.flags[Dirty::Scissors] = true;

    if (!regs.viewport_scale_offset_enabled) {
        float x = static_cast<float>(regs.surface_clip.x);
        float y = static_cast<float>(regs.surface_clip.y);
        float width = (std::max)(1.0f, static_cast<float>(regs.surface_clip.width));
        float height = (std::max)(1.0f, static_cast<float>(regs.surface_clip.height));
        if (regs.window_origin.mode != Maxwell::WindowOrigin::Mode::UpperLeft) {
            y += height;
            height = -height;
        }
        VkViewport viewport{
            .x = x,
            .y = y,
            .width = width,
            .height = height,
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        scheduler.Record([this, viewport](vk::CommandBuffer cmdbuf) {
            const u32 num_viewports = std::min<u32>(device.GetMaxViewports(), Maxwell::NumViewports);
            std::array<VkViewport, Maxwell::NumViewports> viewport_list{};
            viewport_list.fill(viewport);
            const vk::Span<VkViewport> viewports(viewport_list.data(), num_viewports);
            cmdbuf.SetViewport(0, viewports);
        });
        return;
    }
    const bool is_rescaling{texture_cache.IsRescaling()};
    const float scale = is_rescaling ? Settings::values.resolution_info.up_factor : 1.0f;
    const std::array viewport_list{
        GetViewportState(device, regs, 0, scale),  GetViewportState(device, regs, 1, scale),
        GetViewportState(device, regs, 2, scale),  GetViewportState(device, regs, 3, scale),
        GetViewportState(device, regs, 4, scale),  GetViewportState(device, regs, 5, scale),
        GetViewportState(device, regs, 6, scale),  GetViewportState(device, regs, 7, scale),
        GetViewportState(device, regs, 8, scale),  GetViewportState(device, regs, 9, scale),
        GetViewportState(device, regs, 10, scale), GetViewportState(device, regs, 11, scale),
        GetViewportState(device, regs, 12, scale), GetViewportState(device, regs, 13, scale),
        GetViewportState(device, regs, 14, scale), GetViewportState(device, regs, 15, scale),
    };
    scheduler.Record([this, viewport_list](vk::CommandBuffer cmdbuf) {
        const u32 num_viewports = std::min<u32>(device.GetMaxViewports(), Maxwell::NumViewports);
        const vk::Span<VkViewport> viewports(viewport_list.data(), num_viewports);
        cmdbuf.SetViewport(0, viewports);
    });
}

void RasterizerVulkan::UpdateScissorsState(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchScissors()) {
        return;
    }
    if (!regs.viewport_scale_offset_enabled) {
        u32 x = regs.surface_clip.x;
        u32 y = regs.surface_clip.y;
        u32 width = (std::max)(1u, static_cast<u32>(regs.surface_clip.width));
        u32 height = (std::max)(1u, static_cast<u32>(regs.surface_clip.height));
        if (regs.window_origin.mode != Maxwell::WindowOrigin::Mode::UpperLeft) {
            y = regs.surface_clip.height - (y + height);
        }
        VkRect2D scissor{};
        scissor.offset.x = static_cast<int32_t>(x);
        scissor.offset.y = static_cast<int32_t>(y);
        scissor.extent.width  = width;
        scissor.extent.height = height;
        scheduler.Record([this, scissor](vk::CommandBuffer cmdbuf) {
            const u32 num_scissors = std::min<u32>(device.GetMaxViewports(), Maxwell::NumViewports);
            std::array<VkRect2D, Maxwell::NumViewports> scissor_list{};
            scissor_list.fill(scissor);
            const vk::Span<VkRect2D> scissors(scissor_list.data(), num_scissors);
            cmdbuf.SetScissor(0, scissors);
        });
        return;
    }
    u32 up_scale = 1;
    u32 down_shift = 0;
    if (texture_cache.IsRescaling()) {
        up_scale = Settings::values.resolution_info.up_scale;
        down_shift = Settings::values.resolution_info.down_shift;
    }
    const std::array scissor_list{
        GetScissorState(regs, 0, up_scale, down_shift),
        GetScissorState(regs, 1, up_scale, down_shift),
        GetScissorState(regs, 2, up_scale, down_shift),
        GetScissorState(regs, 3, up_scale, down_shift),
        GetScissorState(regs, 4, up_scale, down_shift),
        GetScissorState(regs, 5, up_scale, down_shift),
        GetScissorState(regs, 6, up_scale, down_shift),
        GetScissorState(regs, 7, up_scale, down_shift),
        GetScissorState(regs, 8, up_scale, down_shift),
        GetScissorState(regs, 9, up_scale, down_shift),
        GetScissorState(regs, 10, up_scale, down_shift),
        GetScissorState(regs, 11, up_scale, down_shift),
        GetScissorState(regs, 12, up_scale, down_shift),
        GetScissorState(regs, 13, up_scale, down_shift),
        GetScissorState(regs, 14, up_scale, down_shift),
        GetScissorState(regs, 15, up_scale, down_shift),
    };
    scheduler.Record([this, scissor_list](vk::CommandBuffer cmdbuf) {
        const u32 num_scissors = std::min<u32>(device.GetMaxViewports(), Maxwell::NumViewports);
        const vk::Span<VkRect2D> scissors(scissor_list.data(), num_scissors);
        cmdbuf.SetScissor(0, scissors);
    });
}

void RasterizerVulkan::UpdateDepthBias(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchDepthBias()) {
        return;
    }
    float units = regs.depth_bias / 2.0f;
    const bool is_d24 = regs.zeta.format == Tegra::DepthFormat::Z24_UNORM_S8_UINT ||
                        regs.zeta.format == Tegra::DepthFormat::X8Z24_UNORM ||
                        regs.zeta.format == Tegra::DepthFormat::S8Z24_UNORM ||
                        regs.zeta.format == Tegra::DepthFormat::V8Z24_UNORM;

    if (is_d24 && !device.SupportsD24DepthBuffer()) {
        static constexpr const size_t length = sizeof(NEEDS_D24) / sizeof(NEEDS_D24[0]);

        static constexpr const u64* start = NEEDS_D24;
        static constexpr const u64* end = NEEDS_D24 + length;

        const u64* it = std::find(start, end, program_id);

        if (it != end) {
            // the base formulas can be obtained from here:
            //   https://docs.microsoft.com/en-us/windows/win32/direct3d11/d3d10-graphics-programming-guide-output-merger-stage-depth-bias
            const double rescale_factor =
                static_cast<double>(1ULL << (32 - 24)) / (static_cast<double>(0x1.ep+127));
            units = static_cast<float>(static_cast<double>(units) * rescale_factor);
        }
    }

    scheduler.Record([constant = units, clamp = regs.depth_bias_clamp,
                      factor = regs.slope_scale_depth_bias, this](vk::CommandBuffer cmdbuf) {
        if (device.IsExtDepthBiasControlSupported()) {
            static VkDepthBiasRepresentationInfoEXT bias_info{
                .sType = VK_STRUCTURE_TYPE_DEPTH_BIAS_REPRESENTATION_INFO_EXT,
                .pNext = nullptr,
                .depthBiasRepresentation =
                    VK_DEPTH_BIAS_REPRESENTATION_LEAST_REPRESENTABLE_VALUE_FORCE_UNORM_EXT,
                .depthBiasExact = VK_FALSE,
            };

            cmdbuf.SetDepthBias(constant, clamp, factor, &bias_info);
        } else {
            cmdbuf.SetDepthBias(constant, clamp, factor);
        }
    });
}

void RasterizerVulkan::UpdateBlendConstants(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchBlendConstants()) {
        return;
    }
    const std::array blend_color = {regs.blend_color.r, regs.blend_color.g, regs.blend_color.b,
                                    regs.blend_color.a};
    scheduler.Record(
        [blend_color](vk::CommandBuffer cmdbuf) { cmdbuf.SetBlendConstants(blend_color.data()); });
}

void RasterizerVulkan::UpdateDepthBounds(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchDepthBounds()) {
        return;
    }
    scheduler.Record([min = regs.depth_bounds[0], max = regs.depth_bounds[1]](
                         vk::CommandBuffer cmdbuf) { cmdbuf.SetDepthBounds(min, max); });
}

void RasterizerVulkan::UpdateStencilFaces(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchStencilProperties()) {
        return;
    }
    bool update_references = state_tracker.TouchStencilReference();
    bool update_write_mask = state_tracker.TouchStencilWriteMask();
    bool update_compare_masks = state_tracker.TouchStencilCompare();
    if (state_tracker.TouchStencilSide(regs.stencil_two_side_enable != 0)) {
        update_references = true;
        update_write_mask = true;
        update_compare_masks = true;
    }
    if (update_references) {
        [&]() {
            bool changed;
            if (regs.stencil_two_side_enable) {
                const bool front_changed =
                    state_tracker.CheckStencilReferenceFront(regs.stencil_front_ref);
                const bool back_changed =
                    state_tracker.CheckStencilReferenceBack(regs.stencil_back_ref);
                changed = front_changed || back_changed;
            } else {
                const bool front_changed =
                    state_tracker.CheckStencilReferenceFront(regs.stencil_front_ref);
                const bool back_changed =
                    state_tracker.CheckStencilReferenceBack(regs.stencil_front_ref);
                changed = front_changed || back_changed;
            }
            if (!changed) {
                return;
            }
            scheduler.Record([front_ref = regs.stencil_front_ref, back_ref = regs.stencil_back_ref,
                              two_sided = regs.stencil_two_side_enable](vk::CommandBuffer cmdbuf) {
                const bool set_back = two_sided && front_ref != back_ref;
                // Front face
                cmdbuf.SetStencilReference(set_back ? VK_STENCIL_FACE_FRONT_BIT
                                                    : VK_STENCIL_FACE_FRONT_AND_BACK,
                                           front_ref);
                if (set_back) {
                    cmdbuf.SetStencilReference(VK_STENCIL_FACE_BACK_BIT, back_ref);
                }
            });
        }();
    }
    if (update_write_mask) {
        [&]() {
            bool changed;
            if (regs.stencil_two_side_enable) {
                const bool front_changed =
                    state_tracker.CheckStencilWriteMaskFront(regs.stencil_front_mask);
                const bool back_changed =
                    state_tracker.CheckStencilWriteMaskBack(regs.stencil_back_mask);
                changed = front_changed || back_changed;
            } else {
                const bool front_changed =
                    state_tracker.CheckStencilWriteMaskFront(regs.stencil_front_mask);
                const bool back_changed =
                    state_tracker.CheckStencilWriteMaskBack(regs.stencil_front_mask);
                changed = front_changed || back_changed;
            }
            if (!changed) {
                return;
            }
            scheduler.Record([front_write_mask = regs.stencil_front_mask,
                              back_write_mask = regs.stencil_back_mask,
                              two_sided = regs.stencil_two_side_enable](vk::CommandBuffer cmdbuf) {
                const bool set_back = two_sided && front_write_mask != back_write_mask;
                // Front face
                cmdbuf.SetStencilWriteMask(set_back ? VK_STENCIL_FACE_FRONT_BIT
                                                    : VK_STENCIL_FACE_FRONT_AND_BACK,
                                           front_write_mask);
                if (set_back) {
                    cmdbuf.SetStencilWriteMask(VK_STENCIL_FACE_BACK_BIT, back_write_mask);
                }
            });
        }();
    }
    if (update_compare_masks) {
        [&]() {
            bool changed;
            if (regs.stencil_two_side_enable) {
                const bool front_changed =
                    state_tracker.CheckStencilCompareMaskFront(regs.stencil_front_func_mask);
                const bool back_changed =
                    state_tracker.CheckStencilCompareMaskBack(regs.stencil_back_func_mask);
                changed = front_changed || back_changed;
            } else {
                const bool front_changed =
                    state_tracker.CheckStencilCompareMaskFront(regs.stencil_front_func_mask);
                const bool back_changed =
                    state_tracker.CheckStencilCompareMaskBack(regs.stencil_front_func_mask);
                changed = front_changed || back_changed;
            }
            if (!changed) {
                return;
            }
            scheduler.Record([front_test_mask = regs.stencil_front_func_mask,
                              back_test_mask = regs.stencil_back_func_mask,
                              two_sided = regs.stencil_two_side_enable](vk::CommandBuffer cmdbuf) {
                const bool set_back = two_sided && front_test_mask != back_test_mask;
                // Front face
                cmdbuf.SetStencilCompareMask(set_back ? VK_STENCIL_FACE_FRONT_BIT
                                                      : VK_STENCIL_FACE_FRONT_AND_BACK,
                                             front_test_mask);
                if (set_back) {
                    cmdbuf.SetStencilCompareMask(VK_STENCIL_FACE_BACK_BIT, back_test_mask);
                }
            });
        }();
    }
    state_tracker.ClearStencilReset();
}

void RasterizerVulkan::UpdateLineWidth(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchLineWidth()) {
        return;
    }
    const float width =
        regs.line_anti_alias_enable ? regs.line_width_smooth : regs.line_width_aliased;
    scheduler.Record([width](vk::CommandBuffer cmdbuf) { cmdbuf.SetLineWidth(width); });
}

void RasterizerVulkan::UpdateCullMode(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchCullMode()) {
        return;
    }
    scheduler.Record([enabled = regs.gl_cull_test_enabled,
                      cull_face = regs.gl_cull_face](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetCullModeEXT(enabled ? MaxwellToVK::CullFace(cull_face) : VK_CULL_MODE_NONE);
    });
}

void RasterizerVulkan::UpdateDepthBoundsTestEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchDepthBoundsTestEnable()) {
        return;
    }
    bool enabled = regs.depth_bounds_enable;
    if (enabled && !device.IsDepthBoundsSupported()) {
        LOG_WARNING(Render_Vulkan, "Depth bounds is enabled but not supported");
        enabled = false;
    }
    scheduler.Record([enable = enabled](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetDepthBoundsTestEnableEXT(enable);
    });
}

void RasterizerVulkan::UpdateDepthTestEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchDepthTestEnable()) {
        return;
    }
    scheduler.Record([enable = regs.depth_test_enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetDepthTestEnableEXT(enable);
    });
}

void RasterizerVulkan::UpdateDepthWriteEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchDepthWriteEnable()) {
        return;
    }
    scheduler.Record([enable = regs.depth_write_enabled](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetDepthWriteEnableEXT(enable);
    });
}

void RasterizerVulkan::UpdatePrimitiveRestartEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchPrimitiveRestartEnable()) {
        return;
    }

    bool enable = regs.primitive_restart.enabled != 0;
    if (device.IsMoltenVK()) {
        enable = true;
    } else if (enable) {
        const auto topology = MaxwellToVK::PrimitiveTopology(device, maxwell3d->draw_manager.draw_state.topology);
        enable = IsPrimitiveRestartSupported(device, topology);
    }

    scheduler.Record([enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetPrimitiveRestartEnableEXT(enable);
    });
}

void RasterizerVulkan::UpdateRasterizerDiscardEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchRasterizerDiscardEnable()) {
        return;
    }
    scheduler.Record([disable = regs.rasterize_enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetRasterizerDiscardEnableEXT(disable == 0);
    });
}

void RasterizerVulkan::UpdateConservativeRasterizationMode(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchConservativeRasterizationMode()) {
        return;
    }

    if (!device.SupportsDynamicState3ConservativeRasterizationMode()) {
        return;
    }

    scheduler.Record([enable = regs.conservative_raster_enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetConservativeRasterizationModeEXT(
            enable ? VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT
                   : VK_CONSERVATIVE_RASTERIZATION_MODE_DISABLED_EXT);
    });
}

void RasterizerVulkan::UpdateLineStippleEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchLineStippleEnable()) {
        return;
    }

    if (!device.SupportsDynamicState3LineStippleEnable()) {
        return;
    }

    scheduler.Record([enable = regs.line_stipple_enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetLineStippleEnableEXT(enable);
    });
}

void RasterizerVulkan::UpdateLineRasterizationMode(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!device.IsExtLineRasterizationSupported()) {
        return;
    }
    if (!state_tracker.TouchLineRasterizationMode()) {
        return;
    }

    if (!device.SupportsDynamicState3LineRasterizationMode()) {
        static std::once_flag warn_missing_rect;
        std::call_once(warn_missing_rect, [] {
            LOG_WARNING(Render_Vulkan,
                        "Driver lacks rectangular line rasterization support; skipping dynamic "
                        "line state updates");
        });
        return;
    }

    const bool wants_smooth = regs.line_anti_alias_enable != 0;
    VkLineRasterizationModeEXT mode = VK_LINE_RASTERIZATION_MODE_RECTANGULAR_EXT;
    if (wants_smooth) {
        if (device.SupportsSmoothLines()) {
            mode = VK_LINE_RASTERIZATION_MODE_RECTANGULAR_SMOOTH_EXT;
        } else {
            static std::once_flag warn_missing_smooth;
            std::call_once(warn_missing_smooth, [] {
                LOG_WARNING(Render_Vulkan,
                            "Line anti-aliasing requested but smoothLines feature unavailable; "
                            "using rectangular rasterization");
            });
        }
    }
    scheduler.Record([mode](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetLineRasterizationModeEXT(mode);
    });
}

void RasterizerVulkan::UpdateDepthBiasEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchDepthBiasEnable()) {
        return;
    }
    constexpr size_t POINT = 0;
    constexpr size_t LINE = 1;
    constexpr size_t POLYGON = 2;
    static constexpr std::array POLYGON_OFFSET_ENABLE_LUT = {
        POINT,   // Points
        LINE,    // Lines
        LINE,    // LineLoop
        LINE,    // LineStrip
        POLYGON, // Triangles
        POLYGON, // TriangleStrip
        POLYGON, // TriangleFan
        POLYGON, // Quads
        POLYGON, // QuadStrip
        POLYGON, // Polygon
        LINE,    // LinesAdjacency
        LINE,    // LineStripAdjacency
        POLYGON, // TrianglesAdjacency
        POLYGON, // TriangleStripAdjacency
        POLYGON, // Patches
    };
    const std::array enabled_lut{
        regs.polygon_offset_point_enable,
        regs.polygon_offset_line_enable,
        regs.polygon_offset_fill_enable,
    };
    const u32 topology_index = u32(maxwell3d->draw_manager.draw_state.topology);
    const u32 enable = enabled_lut[POLYGON_OFFSET_ENABLE_LUT[topology_index]];
    scheduler.Record([enable](vk::CommandBuffer cmdbuf) { cmdbuf.SetDepthBiasEnableEXT(enable != 0); });
}

void RasterizerVulkan::UpdateLogicOpEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchLogicOpEnable()) {
        return;
    }
    if (!device.SupportsDynamicState3LogicOpEnable()) {
        return;
    }
    scheduler.Record([enable = regs.logic_op.enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetLogicOpEnableEXT(enable != 0);
    });
}

void RasterizerVulkan::UpdateDepthClampEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchDepthClampEnable()) {
        return;
    }
    if (!device.SupportsDynamicState3DepthClampEnable()) {
        return;
    }
    bool is_enabled = !(regs.viewport_clip_control.geometry_clip ==
                            Maxwell::ViewportClipControl::GeometryClip::Passthrough ||
                        regs.viewport_clip_control.geometry_clip ==
                            Maxwell::ViewportClipControl::GeometryClip::FrustumXYZ ||
                        regs.viewport_clip_control.geometry_clip ==
                            Maxwell::ViewportClipControl::GeometryClip::FrustumZ);
    scheduler.Record(
        [is_enabled](vk::CommandBuffer cmdbuf) { cmdbuf.SetDepthClampEnableEXT(is_enabled); });
}

void RasterizerVulkan::UpdateAlphaToCoverageEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchAlphaToCoverageEnable()) {
        return;
    }
    if (!device.SupportsDynamicState3AlphaToCoverageEnable()) {
        return;
    }
    GraphicsPipeline* const pipeline = pipeline_cache.CurrentGraphicsPipeline();
    const bool enable = pipeline != nullptr && pipeline->SupportsAlphaToCoverage() &&
                        regs.anti_alias_alpha_control.alpha_to_coverage != 0;
    scheduler.Record([enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetAlphaToCoverageEnableEXT(enable ? VK_TRUE : VK_FALSE);
    });
}

void RasterizerVulkan::UpdateAlphaToOneEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchAlphaToOneEnable()) {
        return;
    }
    if (!device.SupportsDynamicState3AlphaToOneEnable()) {
        static std::once_flag warn_alpha_to_one;
        std::call_once(warn_alpha_to_one, [] {
            LOG_WARNING(Render_Vulkan,
                        "Alpha-to-one is not supported on this device; forcing it disabled");
        });
        return;
    }
    GraphicsPipeline* const pipeline = pipeline_cache.CurrentGraphicsPipeline();
    const bool enable = pipeline != nullptr && pipeline->SupportsAlphaToOne() &&
                        regs.anti_alias_alpha_control.alpha_to_one != 0;
    scheduler.Record([enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetAlphaToOneEnableEXT(enable ? VK_TRUE : VK_FALSE);
    });
}

void RasterizerVulkan::UpdateDepthCompareOp(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchDepthCompareOp()) {
        return;
    }
    scheduler.Record([func = regs.depth_test_func](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetDepthCompareOpEXT(MaxwellToVK::ComparisonOp(func));
    });
}

void RasterizerVulkan::UpdateFrontFace(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchFrontFace()) {
        return;
    }

    VkFrontFace front_face = MaxwellToVK::FrontFace(regs.gl_front_face);
    if (regs.window_origin.flip_y != 0) {
        front_face = front_face == VK_FRONT_FACE_CLOCKWISE ? VK_FRONT_FACE_COUNTER_CLOCKWISE
                                                           : VK_FRONT_FACE_CLOCKWISE;
    }
    scheduler.Record(
        [front_face](vk::CommandBuffer cmdbuf) { cmdbuf.SetFrontFaceEXT(front_face); });
}

void RasterizerVulkan::UpdateStencilOp(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchStencilOp()) {
        return;
    }
    const Maxwell::StencilOp::Op fail = regs.stencil_front_op.fail;
    const Maxwell::StencilOp::Op zfail = regs.stencil_front_op.zfail;
    const Maxwell::StencilOp::Op zpass = regs.stencil_front_op.zpass;
    const Maxwell::ComparisonOp compare = regs.stencil_front_op.func;
    if (regs.stencil_two_side_enable) {
        // Separate stencil op per face
        const Maxwell::StencilOp::Op back_fail = regs.stencil_back_op.fail;
        const Maxwell::StencilOp::Op back_zfail = regs.stencil_back_op.zfail;
        const Maxwell::StencilOp::Op back_zpass = regs.stencil_back_op.zpass;
        const Maxwell::ComparisonOp back_compare = regs.stencil_back_op.func;
        scheduler.Record([fail, zfail, zpass, compare, back_fail, back_zfail, back_zpass,
                          back_compare](vk::CommandBuffer cmdbuf) {
            cmdbuf.SetStencilOpEXT(VK_STENCIL_FACE_FRONT_BIT, MaxwellToVK::StencilOp(fail),
                                   MaxwellToVK::StencilOp(zpass), MaxwellToVK::StencilOp(zfail),
                                   MaxwellToVK::ComparisonOp(compare));
            cmdbuf.SetStencilOpEXT(VK_STENCIL_FACE_BACK_BIT, MaxwellToVK::StencilOp(back_fail),
                                   MaxwellToVK::StencilOp(back_zpass),
                                   MaxwellToVK::StencilOp(back_zfail),
                                   MaxwellToVK::ComparisonOp(back_compare));
        });
    } else {
        // Front face defines the stencil op of both faces
        scheduler.Record([fail, zfail, zpass, compare](vk::CommandBuffer cmdbuf) {
            cmdbuf.SetStencilOpEXT(VK_STENCIL_FACE_FRONT_AND_BACK, MaxwellToVK::StencilOp(fail),
                                   MaxwellToVK::StencilOp(zpass), MaxwellToVK::StencilOp(zfail),
                                   MaxwellToVK::ComparisonOp(compare));
        });
    }
}

void RasterizerVulkan::UpdateLogicOp(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchLogicOp()) {
        return;
    }
    const auto op_value = static_cast<u32>(regs.logic_op.op);
    auto op = op_value >= 0x1500 && op_value < 0x1510 ? static_cast<VkLogicOp>(op_value - 0x1500)
                                                      : VK_LOGIC_OP_NO_OP;
    scheduler.Record([op](vk::CommandBuffer cmdbuf) { cmdbuf.SetLogicOpEXT(op); });
}

void RasterizerVulkan::UpdateBlending(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchBlending()) {
        return;
    }

    if (state_tracker.TouchColorMask()) {
        std::array<VkColorComponentFlags, Maxwell::NumRenderTargets> setup_masks{};
        for (size_t index = 0; index < Maxwell::NumRenderTargets; index++) {
            const auto& mask = regs.color_mask[regs.color_mask_common ? 0 : index];
            auto& current = setup_masks[index];
            if (mask.R) {
                current |= VK_COLOR_COMPONENT_R_BIT;
            }
            if (mask.G) {
                current |= VK_COLOR_COMPONENT_G_BIT;
            }
            if (mask.B) {
                current |= VK_COLOR_COMPONENT_B_BIT;
            }
            if (mask.A) {
                current |= VK_COLOR_COMPONENT_A_BIT;
            }
        }
        scheduler.Record([setup_masks](vk::CommandBuffer cmdbuf) {
            cmdbuf.SetColorWriteMaskEXT(0, setup_masks);
        });
    }

    if (state_tracker.TouchBlendEnable()) {
        std::array<VkBool32, Maxwell::NumRenderTargets> setup_enables{};
        for (size_t index = 0; index < Maxwell::NumRenderTargets; index++) {
            bool is_integer = false;
            if (regs.rt[index].format != Tegra::RenderTargetFormat::NONE) {
                const auto format =
                    VideoCore::Surface::PixelFormatFromRenderTargetFormat(regs.rt[index].format);
                is_integer = IsPixelFormatInteger(format);
            }
            setup_enables[index] =
                (!is_integer && regs.blend.enable[index] != 0) ? VK_TRUE : VK_FALSE;
        }
        scheduler.Record([setup_enables](vk::CommandBuffer cmdbuf) {
            cmdbuf.SetColorBlendEnableEXT(0, setup_enables);
        });
    }

    if (state_tracker.TouchBlendEquations()) {
        std::array<VkColorBlendEquationEXT, Maxwell::NumRenderTargets> setup_blends{};

        const auto blend_setup = [&](auto& host_blend, const auto& guest_blend) {
            host_blend.srcColorBlendFactor = MaxwellToVK::BlendFactor(guest_blend.color_source);
            host_blend.dstColorBlendFactor = MaxwellToVK::BlendFactor(guest_blend.color_dest);
            host_blend.colorBlendOp = MaxwellToVK::BlendEquation(guest_blend.color_op);
            host_blend.srcAlphaBlendFactor = MaxwellToVK::BlendFactor(guest_blend.alpha_source);
            host_blend.dstAlphaBlendFactor = MaxwellToVK::BlendFactor(guest_blend.alpha_dest);
            host_blend.alphaBlendOp = MaxwellToVK::BlendEquation(guest_blend.alpha_op);
        };

        // Single blend equation for all targets
        if (!regs.blend_per_target_enabled) {
            // Temporary workaround for games that use iterated blending
            if (regs.iterated_blend.enable && Settings::values.use_squashed_iterated_blend) {
                setup_blends[0].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
                setup_blends[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
                setup_blends[0].colorBlendOp = VK_BLEND_OP_ADD;
                setup_blends[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
                setup_blends[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
                setup_blends[0].alphaBlendOp = VK_BLEND_OP_ADD;
            } else {
                blend_setup(setup_blends[0], regs.blend);
            }

            // Copy first blend state to all other targets
            for (size_t index = 1; index < Maxwell::NumRenderTargets; index++) {
                setup_blends[index] = setup_blends[0];
            }
        } else {
            // Per-target blending
            for (size_t index = 0; index < Maxwell::NumRenderTargets; index++) {
                blend_setup(setup_blends[index], regs.blend_per_target[index]);
            }
        }

        scheduler.Record([setup_blends](vk::CommandBuffer cmdbuf) {
            cmdbuf.SetColorBlendEquationEXT(0, setup_blends);
        });
    }
}

void RasterizerVulkan::UpdateColorWriteEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchColorMask()) {
        return;
    }
    std::array<VkBool32, Maxwell::NumRenderTargets> setup_enables{};
    for (size_t index = 0; index < Maxwell::NumRenderTargets; index++) {
        const auto& mask = regs.color_mask[regs.color_mask_common ? 0 : index];
        setup_enables[index] = (mask.R || mask.G || mask.B || mask.A) ? VK_TRUE : VK_FALSE;
    }
    scheduler.Record([setup_enables](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetColorWriteEnableEXT(setup_enables);
    });
}

void RasterizerVulkan::UpdateStencilTestEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchStencilTestEnable()) {
        return;
    }
    scheduler.Record([enable = regs.stencil_enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetStencilTestEnableEXT(enable);
    });
}

void RasterizerVulkan::UpdateVertexInput(Tegra::Engines::Maxwell3D::Regs& regs) {
    auto& dirty{maxwell3d->dirty.flags};
    const bool vertex_input_dirty = dirty[Dirty::VertexInput];
    const bool vertex_buffers_dirty = dirty[VideoCommon::Dirty::VertexBuffers];
    if (!vertex_input_dirty && !vertex_buffers_dirty) {
        return;
    }
    dirty[Dirty::VertexInput] = false;

    boost::container::static_vector<VkVertexInputBindingDescription2EXT, 32> bindings;
    boost::container::static_vector<VkVertexInputAttributeDescription2EXT, 32> attributes;

    const u32 max_attributes =
        static_cast<u32>(std::min<size_t>(Maxwell::NumVertexAttributes,
                                          device.GetMaxVertexInputAttributes()));
    const u32 max_bindings =
        static_cast<u32>(std::min<size_t>(Maxwell::NumVertexArrays,
                                          device.GetMaxVertexInputBindings()));


    for (u32 index = 0; index < max_attributes; ++index) {
        const Maxwell::VertexAttribute attribute{regs.vertex_attrib_format[index]};
        const u32 binding{attribute.buffer};
        if (attribute.constant || binding >= max_bindings) {
            continue;
        }
        attributes.push_back({
            .sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
            .pNext = nullptr,
            .location = index,
            .binding = binding,
            .format = MaxwellToVK::VertexFormat(device, attribute.type, attribute.size),
            .offset = attribute.offset,
        });
    }

    for (u32 binding = 0; binding < max_bindings; ++binding) {
        const auto& input_binding{regs.vertex_streams[binding]};
        const bool is_instanced{regs.vertex_stream_instances.IsInstancingEnabled(binding)};
        bindings.push_back({
            .sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT,
            .pNext = nullptr,
            .binding = binding,
            .stride = input_binding.stride,
            .inputRate = is_instanced ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX,
            .divisor = is_instanced ? input_binding.frequency : 1,
        });
    }

    for (size_t index = 0; index < Maxwell::NumVertexAttributes; ++index) {
        dirty[Dirty::VertexAttribute0 + index] = false;
    }
    for (size_t index = 0; index < Maxwell::NumVertexArrays; ++index) {
        dirty[Dirty::VertexBinding0 + index] = false;
    }

    scheduler.Record([bindings, attributes](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetVertexInputEXT(bindings, attributes);
    });
}

void RasterizerVulkan::InitializeChannel(Tegra::Control::ChannelState& channel) {
    CreateChannel(channel);
    {
        std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
        texture_cache.CreateChannel(channel);
        buffer_cache.CreateChannel(channel);
    }
    pipeline_cache.CreateChannel(channel);
    query_cache.CreateChannel(channel);
    state_tracker.SetupTables(channel);
}

void RasterizerVulkan::BindChannel(Tegra::Control::ChannelState& channel) {
    const s32 channel_id = channel.bind_id;
    BindToChannel(channel_id);
    {
        std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
        texture_cache.BindToChannel(channel_id);
        buffer_cache.BindToChannel(channel_id);
    }
    pipeline_cache.BindToChannel(channel_id);
    query_cache.BindToChannel(channel_id);
    state_tracker.ChangeChannel(channel);
    state_tracker.InvalidateState();
}

void RasterizerVulkan::ReleaseChannel(s32 channel_id) {
    state_tracker.ReleaseChannel(channel_id);
    EraseChannel(channel_id);
    {
        std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
        texture_cache.EraseChannel(channel_id);
        buffer_cache.EraseChannel(channel_id);
    }
    pipeline_cache.EraseChannel(channel_id);
    query_cache.EraseChannel(channel_id);
}

} // namespace Vulkan
