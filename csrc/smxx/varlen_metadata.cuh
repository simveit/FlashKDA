#pragma once

#include "../fwd.h"

#include <cub/cub.cuh>

template <int CHUNK>
__global__ __launch_bounds__(kVarlenMetadataThreads) void _flash_kda_count_varlen_chunks(
    int64_t const* cu_seqlens,
    int32_t* chunk_offsets,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int seq_len = int(cu_seqlens[idx + 1] - cu_seqlens[idx]);
        chunk_offsets[idx] = (seq_len + CHUNK - 1) / CHUNK;
    }
    if (idx == N) {
        chunk_offsets[idx] = 0;
    }
}

__global__ __launch_bounds__(kVarlenMetadataThreads) void _flash_kda_fill_varlen_metadata(
    VarlenMetadata metadata,
    int N
) {
    int seq_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (seq_id < N) {
        int begin_chunk = metadata.chunk_offsets[seq_id];
        int end_chunk = metadata.chunk_offsets[seq_id + 1];
        for (int chunk = begin_chunk; chunk < end_chunk; ++chunk) {
            metadata.chunk_indices[chunk] = make_int2(seq_id, chunk - begin_chunk);
        }
    }
}

template <int CHUNK>
void launch_varlen_metadata(
    int64_t const* cu_seqlens,
    VarlenMetadata metadata,
    int N,
    cudaStream_t stream
) {
    if (!metadata.enabled()) {
        return;
    }

    int blocks = (N + 1 + kVarlenMetadataThreads - 1) / kVarlenMetadataThreads;
    _flash_kda_count_varlen_chunks<CHUNK><<<blocks, kVarlenMetadataThreads, 0, stream>>>(
        cu_seqlens, metadata.chunk_offsets, N);

    void* temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    // Follow CUB's two-phase DeviceScan API: query temp storage, allocate it,
    // then run the in-place exclusive sum.
    cub::DeviceScan::ExclusiveSum(
        temp_storage, temp_storage_bytes, metadata.chunk_offsets,
        metadata.chunk_offsets, N + 1, stream);
    cudaMalloc(&temp_storage, temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(
        temp_storage, temp_storage_bytes, metadata.chunk_offsets,
        metadata.chunk_offsets, N + 1, stream);
    cudaFree(temp_storage);

    blocks = (N + kVarlenMetadataThreads - 1) / kVarlenMetadataThreads;
    _flash_kda_fill_varlen_metadata<<<blocks, kVarlenMetadataThreads, 0, stream>>>(
        metadata, N);
}
