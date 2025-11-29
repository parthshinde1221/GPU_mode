# Old CMake - just add two exuctables if you wnat to make multiple files
# set(TARGET matmul)

# add_executable(${TARGET}
#     matmul.cu
# )

# set_target_properties(${TARGET} PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON
# )

# Glob all .cu files in this directory
file(GLOB REDUCTION_SOURCES *.cu)

foreach(src ${REDUCTION_SOURCES})
    # Get the filename without extension -> becomes the target name
    get_filename_component(target ${src} NAME_WE)

    message(STATUS "Adding reduction target: ${target} from ${src}")

    add_executable(${target}
        ${src}
    )

    set_target_properties(${target} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endforeach()
