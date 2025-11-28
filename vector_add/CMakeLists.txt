# set(TARGET vec_add)

# add_executable(${TARGET}
#     vec_add.cu
# )

# set_target_properties(${TARGET} PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON
# )


file(GLOB VECADD_SOURCES *.cu)

foreach(src ${VECADD_SOURCES})
    get_filename_component(target ${src} NAME_WE)
    message(STATUS "Adding vec_add target: ${target} from ${src}")

    add_executable(${target}
        ${src}
    )

    set_target_properties(${target} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endforeach()
