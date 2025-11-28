set(TARGET matmul)

add_executable(${TARGET}
    matmul.cu
)

set_target_properties(${TARGET} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
