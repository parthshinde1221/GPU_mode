set(TARGET vec_add)

add_executable(${TARGET}
    vec_add.cu
)

set_target_properties(${TARGET} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
