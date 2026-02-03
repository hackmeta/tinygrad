#include "kittens.cuh"

using namespace kittens;

constexpr int NUM_WORKERS = 4;
constexpr int PIPE_STAGES = 3;

#ifndef FA_B
#define FA_B 16
#endif
#ifndef FA_N
#define FA_N 1024
#endif
#ifndef FA_H
#define FA_H 16
#endif
#ifndef FA_D
#define FA_D 64
#endif

template<int D> constexpr size_t ROWS = 16*(64/D);
template<int D, typename T=bf16, typename L=row_l> using qkvo_tile = rt<T, ROWS<D>, D, L>;
template<int D, typename T=float> using attn_tile = rt<T, ROWS<D>, ROWS<D>>;
template<int D> using shared_tile = st_bf<ROWS<D>, D>;
template<int D> using global_layout = gl<bf16, -1, -1, -1, D>;
template<int D> struct globals { global_layout<D> Qg, Kg, Vg, Og; };

// Flash attention kernel without mask
__launch_bounds__(NUM_WORKERS*WARP_THREADS, 1)
__global__ void attend_ker(bf16 *O_ptr, bf16 *Q_ptr, bf16 *K_ptr, bf16 *V_ptr) {
    constexpr int D = FA_D;
    constexpr int ATTN_B = FA_B;
    constexpr int ATTN_N = FA_N;
    constexpr int ATTN_H = FA_H;
    
    global_layout<D> Qg{Q_ptr, ATTN_B, ATTN_N, ATTN_H, nullptr};
    global_layout<D> Kg{K_ptr, ATTN_B, ATTN_N, ATTN_H, nullptr};
    global_layout<D> Vg{V_ptr, ATTN_B, ATTN_N, ATTN_H, nullptr};
    global_layout<D> Og{O_ptr, ATTN_B, ATTN_N, ATTN_H, nullptr};
    globals<D> g(Qg, Kg, Vg, Og);

    using load_group = kittens::group<2>;
    int loadid = load_group::groupid(), workerid = kittens::warpid();
    constexpr int LOAD_BLOCKS = NUM_WORKERS / load_group::GROUP_WARPS;
    const int batch = blockIdx.z, head = blockIdx.y, q_seq = blockIdx.x * NUM_WORKERS + workerid;

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    shared_tile<D> (&k_smem)[LOAD_BLOCKS][PIPE_STAGES] = al.allocate<shared_tile<D>, LOAD_BLOCKS, PIPE_STAGES>();
    shared_tile<D> (&v_smem)[LOAD_BLOCKS][PIPE_STAGES] = al.allocate<shared_tile<D>, LOAD_BLOCKS, PIPE_STAGES>();
    shared_tile<D> (&qo_smem)[NUM_WORKERS] = reinterpret_cast<shared_tile<D>(&)[NUM_WORKERS]>(k_smem);

    qkvo_tile<D, bf16> q_reg, k_reg;
    qkvo_tile<D, bf16, col_l> v_reg;
    qkvo_tile<D, float> o_reg;
    attn_tile<D, float> att_block;
    attn_tile<D, bf16> att_block_mma;
    typename attn_tile<D, float>::col_vec max_vec_last, max_vec, norm_vec;

    if (q_seq*ROWS<D> < g.Qg.depth()) {
      warp::load<1, false>(qo_smem[workerid], g.Qg, {batch, q_seq, head, 0});
      __syncwarp();
      warp::load(q_reg, qo_smem[workerid]);
    }
    __syncthreads();

    if constexpr(D == 64) q_reg *= __float2bfloat16(0.125f * 1.44269504089f);
    else if constexpr(D == 128) q_reg *= __float2bfloat16(0.08838834764f * 1.44269504089f);

    max_vec = base_types::constants<float>::neg_infty();
    norm_vec = 0.f;
    o_reg = 0.f;

    int kv_blocks = (g.Kg.depth() + LOAD_BLOCKS*ROWS<D>-1) / (LOAD_BLOCKS*ROWS<D>), tic = 0;
    load_group::load_async<1, false>(k_smem[loadid][0], g.Kg, {batch, loadid, head, 0});
    load_group::load_async<1, false>(v_smem[loadid][0], g.Vg, {batch, loadid, head, 0});

    for(auto kv_idx = 0; kv_idx < kv_blocks; kv_idx++, tic=(tic+1)%3) {
        int next_load_idx = (kv_idx+1)*LOAD_BLOCKS + loadid;
        if(next_load_idx*ROWS<D> < g.Kg.depth()) {
            int next_tic = (tic+1)%3;
            load_group::load_async<1, false>(k_smem[loadid][next_tic], g.Kg, {batch, next_load_idx, head, 0});
            load_group::load_async<1, false>(v_smem[loadid][next_tic], g.Vg, {batch, next_load_idx, head, 0});
            load_async_wait<1>();
        }
        else load_async_wait();
        __syncthreads();

        #pragma unroll LOAD_BLOCKS
        for(int subtile = 0; subtile < LOAD_BLOCKS && (kv_idx*LOAD_BLOCKS + subtile)*ROWS<D> < g.Kg.depth(); subtile++) {
          warp::load(k_reg, k_smem[subtile][tic]);
          att_block = 0.f;
          warp::mma<transpose::N, transpose::T>(att_block, q_reg, k_reg, att_block);
          max_vec_last = max_vec;
          max_vec = warp::max<axis::COL>(att_block, max_vec);
          att_block = warp::exp2(att_block - max_vec);
          max_vec_last = warp::exp2(max_vec_last - max_vec);
          norm_vec *= max_vec_last;
          norm_vec = warp::sum<axis::COL>(att_block, norm_vec);
          att_block_mma = att_block;
          warp::load(v_reg, v_smem[subtile][tic]);
          o_reg *= max_vec_last;
          warp::mma<transpose::N, transpose::N>(o_reg, att_block_mma, v_reg, o_reg);
        }
    }

    o_reg /= norm_vec;
    __syncthreads();

    if (q_seq*ROWS<D> < g.Og.depth()) {
      warp::store(qo_smem[workerid], o_reg);
      __syncwarp();
      warp::store<1, false>(g.Og, qo_smem[workerid], {batch, q_seq, head, 0});
    }
}

// Flash attention kernel with causal mask - simpler version
// Process all KV blocks but apply mask
__launch_bounds__(NUM_WORKERS*WARP_THREADS, 1)
__global__ void attend_ker_causal(bf16 *O_ptr, bf16 *Q_ptr, bf16 *K_ptr, bf16 *V_ptr) {
    constexpr int D = FA_D;
    constexpr int ATTN_B = FA_B;
    constexpr int ATTN_N = FA_N;
    constexpr int ATTN_H = FA_H;
    
    global_layout<D> Qg{Q_ptr, ATTN_B, ATTN_N, ATTN_H, nullptr};
    global_layout<D> Kg{K_ptr, ATTN_B, ATTN_N, ATTN_H, nullptr};
    global_layout<D> Vg{V_ptr, ATTN_B, ATTN_N, ATTN_H, nullptr};
    global_layout<D> Og{O_ptr, ATTN_B, ATTN_N, ATTN_H, nullptr};
    globals<D> g(Qg, Kg, Vg, Og);

    using load_group = kittens::group<2>;
    int loadid = load_group::groupid(), workerid = kittens::warpid();
    constexpr int LOAD_BLOCKS = NUM_WORKERS / load_group::GROUP_WARPS;
    const int batch = blockIdx.z, head = blockIdx.y, q_seq = blockIdx.x * NUM_WORKERS + workerid;
    const int q_start_row = q_seq * ROWS<D>;  // Global row offset for this warp's Q tile

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    shared_tile<D> (&k_smem)[LOAD_BLOCKS][PIPE_STAGES] = al.allocate<shared_tile<D>, LOAD_BLOCKS, PIPE_STAGES>();
    shared_tile<D> (&v_smem)[LOAD_BLOCKS][PIPE_STAGES] = al.allocate<shared_tile<D>, LOAD_BLOCKS, PIPE_STAGES>();
    shared_tile<D> (&qo_smem)[NUM_WORKERS] = reinterpret_cast<shared_tile<D>(&)[NUM_WORKERS]>(k_smem);

    qkvo_tile<D, bf16> q_reg, k_reg;
    qkvo_tile<D, bf16, col_l> v_reg;
    qkvo_tile<D, float> o_reg;
    attn_tile<D, float> att_block;
    attn_tile<D, bf16> att_block_mma;
    typename attn_tile<D, float>::col_vec max_vec_last, max_vec, norm_vec;

    if (q_seq*ROWS<D> < g.Qg.depth()) {
      warp::load<1, false>(qo_smem[workerid], g.Qg, {batch, q_seq, head, 0});
      __syncwarp();
      warp::load(q_reg, qo_smem[workerid]);
    }
    __syncthreads();

    if constexpr(D == 64) q_reg *= __float2bfloat16(0.125f * 1.44269504089f);
    else if constexpr(D == 128) q_reg *= __float2bfloat16(0.08838834764f * 1.44269504089f);

    max_vec = base_types::constants<float>::neg_infty();
    norm_vec = 0.f;
    o_reg = 0.f;

    int kv_blocks = (g.Kg.depth() + LOAD_BLOCKS*ROWS<D>-1) / (LOAD_BLOCKS*ROWS<D>), tic = 0;
    load_group::load_async<1, false>(k_smem[loadid][0], g.Kg, {batch, loadid, head, 0});
    load_group::load_async<1, false>(v_smem[loadid][0], g.Vg, {batch, loadid, head, 0});

    for(auto kv_idx = 0; kv_idx < kv_blocks; kv_idx++, tic=(tic+1)%3) {
        int next_load_idx = (kv_idx+1)*LOAD_BLOCKS + loadid;
        if(next_load_idx*ROWS<D> < g.Kg.depth()) {
            int next_tic = (tic+1)%3;
            load_group::load_async<1, false>(k_smem[loadid][next_tic], g.Kg, {batch, next_load_idx, head, 0});
            load_group::load_async<1, false>(v_smem[loadid][next_tic], g.Vg, {batch, next_load_idx, head, 0});
            load_async_wait<1>();
        }
        else load_async_wait();
        __syncthreads();

        #pragma unroll LOAD_BLOCKS
        for(int subtile = 0; subtile < LOAD_BLOCKS && (kv_idx*LOAD_BLOCKS + subtile)*ROWS<D> < g.Kg.depth(); subtile++) {
          int kv_block_idx = kv_idx*LOAD_BLOCKS + subtile;
          int kv_start_col = kv_block_idx * ROWS<D>;  // Global col offset for this KV tile
          
          warp::load(k_reg, k_smem[subtile][tic]);
          att_block = 0.f;
          warp::mma<transpose::N, transpose::T>(att_block, q_reg, k_reg, att_block);
          
          // Apply causal mask: for each (i, j) in tile, mask if (kv_start_col + j) > (q_start_row + i)
          // Using tril with diagonal = q_start_row - kv_start_col
          // tril keeps elements where col <= row + diagonal
          // So: j <= i + (q_start_row - kv_start_col)
          // => kv_start_col + j <= q_start_row + i  ✓
          int diagonal = q_start_row - kv_start_col;
          warp::tril(att_block, att_block, diagonal, base_types::constants<float>::neg_infty());
          
          max_vec_last = max_vec;
          max_vec = warp::max<axis::COL>(att_block, max_vec);
          att_block = warp::exp2(att_block - max_vec);
          max_vec_last = warp::exp2(max_vec_last - max_vec);
          norm_vec *= max_vec_last;
          norm_vec = warp::sum<axis::COL>(att_block, norm_vec);
          att_block_mma = att_block;
          warp::load(v_reg, v_smem[subtile][tic]);
          o_reg *= max_vec_last;
          warp::mma<transpose::N, transpose::N>(o_reg, att_block_mma, v_reg, o_reg);
        }
    }

    o_reg /= norm_vec;
    __syncthreads();

    if (q_seq*ROWS<D> < g.Og.depth()) {
      warp::store(qo_smem[workerid], o_reg);
      __syncwarp();
      warp::store<1, false>(g.Og, qo_smem[workerid], {batch, q_seq, head, 0});
    }
}
