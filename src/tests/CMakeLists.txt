# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

# SPDX-FileCopyrightText: 2018 yuzu Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later

add_executable(tests
    common/bit_field.cpp
    common/cityhash.cpp
    common/container_hash.cpp
    common/fibers.cpp
    common/host_memory.cpp
    common/param_package.cpp
    common/range_map.cpp
    common/ring_buffer.cpp
    common/scratch_buffer.cpp
    common/undefined_fix.cpp
    common/unique_function.cpp
    core/kernel/cpu_topology.cpp
    core/core_timing.cpp
    core/hle/tipc.cpp
    core/hle/native_gpu_characteristics.cpp
    core/hle/native_gpu_address_space.cpp
    core/hle/native_gpu_scheduling_group.cpp
    core/hle/native_gpu_channel.cpp
    core/hle/native_gpu_context.cpp
    core/hle/native_syncpoint_shim.cpp
    core/hle/gpu_memory_attribute.cpp
    core/hle/native_image_layout.cpp
    core/internal_network/network.cpp
    core/loader/deconstructed_rom_directory.cpp
    video_core/memory_tracker.cpp
    video_core/native_compute.cpp
    video_core/turing_macro.cpp
    video_core/native_shader.cpp
    input_common/calibration_configuration_job.cpp
)

create_target_directory_groups(tests)

target_link_libraries(tests PRIVATE common core input_common video_core)
if (TARGET dynarmic::dynarmic)
    target_sources(tests PRIVATE core/arm/code_page_cache.cpp)
    target_sources(tests PRIVATE core/arm/authenticated_return.cpp)
    target_sources(tests PRIVATE core/arm/acquire_load.cpp)
    target_sources(tests PRIVATE core/arm/compare_exchange.cpp)
    target_sources(tests PRIVATE core/arm/atomic_memory_access.cpp)
    target_link_libraries(tests PRIVATE dynarmic::dynarmic)
endif()
target_link_libraries(tests PRIVATE ${PLATFORM_LIBRARIES} Catch2::Catch2WithMain Threads::Threads)

add_test(NAME tests COMMAND tests)

# needed for vma
if (NOT MSVC)
    target_compile_options(tests PRIVATE
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wno-conversion>
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wno-unused-variable>
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wno-unused-parameter>
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wno-missing-field-initializers>)
endif()
