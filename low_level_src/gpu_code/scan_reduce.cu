#ifndef DICOMNODE_SCAN_REDUCE
#define DICOMNODE_SCAN_REDUCE

#include<stdint.h>
#include"../concepts/binary_operator.cu"

enum STATUS_FLAGS: uint8_t {
  STATUS_INVALID=0,
  STATUS_AGGREGATE=1,
  STATUS_PREFIX=2,
};

using Flag = STATUS_FLAGS;

class FlagAnd {
public:
  static __device__ __host__ inline Flag apply(const Flag a, const Flag b) {return Flag(a && b);}
  static __device__ __host__ inline bool equals(const Flag a, const Flag b) { return a == b;}
  static __device__ __host__ inline Flag identity() { return Flag(1);}
  static __device__ __host__ inline Flag remove_volatile(volatile Flag& a) { Flag f = a; return f;}
};

template<typename T>
struct FlagVal {
  T val;
  Flag flag;

  __device__ __host__ inline FlagVal() : val(0), flag(STATUS_INVALID) {}
  __device__ __host__ inline FlagVal(const Flag& _flag, const T& value) {
    val = value;
    flag = _flag;
  }

  __device__ __host__ inline FlagVal(volatile FlagVal& flag_val) {
    val = flag_val.val;
    flag = flag_val.flag;
  }

  __device__ __host__ inline void operator=(const FlagVal& flag_val) volatile {
    val = flag_val.val;
    flag = flag_val.flag;
  }
};


template<typename T, CBinaryOperator<T> OP>
class FlagOp {
public:
  static __device__ __host__ inline FlagVal<T> identity() {return FlagVal<T>(STATUS_AGGREGATE, OP::identity());}
  static __device__ __host__ inline bool equals(const FlagVal<T> a, const FlagVal<T> b){
    return OP::equals(a.val, b.val) && a.flag == b.flag;
  }


  static __device__ __host__ inline FlagVal<T> apply(const FlagVal<T> a, const FlagVal<T> b){
    // This is here for return value optimization
    FlagVal<T> returnStruct;
    if(b.flag == STATUS_PREFIX){
      returnStruct.flag = STATUS_PREFIX;
      returnStruct.val = b.val;
    } else if(a.flag == STATUS_INVALID || b.flag == STATUS_INVALID){
      returnStruct.flag = STATUS_INVALID;
    } else if (a.flag == STATUS_PREFIX) { // b.flag == STATUS_AGGREGATE
      returnStruct.flag = STATUS_PREFIX;
      returnStruct.val = OP::apply(a.val, b.val);
    } else { // (a,b).flag == STATUS_AGGREGATE
      returnStruct.flag = STATUS_AGGREGATE;
      returnStruct.val = OP::apply(a.val, b.val);
    }
    return returnStruct;
  }

  static __device__ __host__ inline FlagVal<T> remove_volatile(volatile FlagVal<T>& flag_val){
    FlagVal<T> fv = flag_val;
    return fv;
  }
};

/**
 * @brief "Reorders" the GPU blocks such that that the first spawned block has
 * id 0. The second have id 1 and so forth. Assumes 1 dimensional block id.
 * All threads in a block must call this function or it deadlocks.
 * uses 4 bytes of shared memory.
 * @param counter_address global address, must be 0 before any calls to this function
 * @return __device__ virtual block id, that should be used in place of blockIdx
 */
__device__ uint32_t get_virtual_block_id(uint32_t* counter_address){
  __shared__ uint32_t address;
  if(threadIdx.x == 0){
    uint32_t local = atomicAdd(counter_address, 1);
    address = local;
  }
  __syncthreads();
  return address;
}

/**
 * @brief Copies data from global memory to shared memory coalleased
 * 
 * @tparam T 
 * @tparam CHUCK 
 * @param dst_shared 
 * @param src_global 
 * @param number_of_elements 
 * @param default_element 
 * @param offset 
 */
template<typename T, uint8_t CHUCK = 1>
__device__ inline void copy_to_shared_memory(volatile T* dst_shared, T* src_global, size_t number_of_elements, T default_element=0, size_t offset=0){
  #pragma unroll
  for(uint8_t i=0; i < CHUCK; i++){
    const uint16_t local_index = threadIdx.x + blockDim.x * i;
    const size_t global_index = local_index + offset;
    T element = default_element;
    if(global_index < number_of_elements){
      element = src_global[global_index];
    }
    dst_shared[local_index] = element;
  }

  __syncthreads();
}

template<typename T, uint8_t CHUCK = 1>
__device__ inline void copy_from_shared_memory(volatile T* dst_global, volatile T* src_shared, size_t number_of_elements, size_t offset=0){
  #pragma unroll
  for(uint8_t i=0; i < CHUCK; i++){
    const uint32_t local_index = threadIdx.x + blockDim.x * i;
    const size_t global_index = local_index + offset;

    if(global_index < number_of_elements){
      dst_global[global_index] = src_shared[local_index];
    }
  }

  __syncthreads();
}

/**
 * @brief 
  * 
  * @tparam T 
  * @tparam OP 
  * @param data 
  * @param idx 
  * @return __device__ 
  */
template<typename T, CBinaryOperator<T> OP>
__device__ inline T scan_inclusive_warp(volatile T* data, const size_t idx){
  const uint_t lane = idx & (WARP_SIZE - 1);

  #pragma unroll
  for(uint8_t i = 0; i < LOG_WARP; i++){
    const uint_t h = 1 << i;

    if(h <= lane){
      data[idx] = OP::apply(data[idx - h], ptr[idx]);
    }
  }

  return data[idx];
}

template<typename T, CBinaryOperator<T> OP>
__device__ inline T scan_inclusive_block(volatile T* shared_memory, const uint32_t index){
    const unsigned int lane   = index & (WARP_SIZE-1);
    const unsigned int warpid = index >> LOG_WARP;

    // 1. perform scan at warp level
    T result = scan_inclusive_warp_kogge_stone<T, OP>(shared_memory, index);
    __syncthreads();

    // 2. Place the end-of-warp results in the first warp.
    if (lane == (WARP_SIZE-1)) {
      shared_memory[warpid] = result;
    }
    __syncthreads();

    // 3. Warp Scan the first warp
    if (warpid == 0) {
      scan_inclusive_warp_kogge_stone<T, OP>(shared_memory, index);
    }
    __syncthreads();

    // 4. accumulate results from previous step;
    if (warpid > 0) {
      result = OP::apply(shared_memory[warpid-1], result);
    }
    __syncthreads();

    shared_memory[index] = result;
    __syncthreads();
    return result;
}

template<typename T, CBinaryOperator<T> OP>
__device__ inline T reduce_inclusive_block(volatile T* shared_memory, const uint32_t index){
  const unsigned int lane   = index & (WARP_SIZE-1);
  const unsigned int warpid = index >> LOG_WARP;

    // 1. perform scan at warp level
    T result = scan_inclusive_warp_kogge_stone<T, OP>(shared_memory, index);
    __syncthreads();

    // 2. Place the end-of-warp results in the first warp.
    if (lane == (WARP_SIZE-1)) {
      shared_memory[warpid] = result;
    }
    __syncthreads();

    // 3. Warp Scan the first warp
    if (warpid == 0) {
      scan_inclusive_warp_kogge_stone<T, OP>(shared_memory, index);
    }
    __syncthreads();

    // 4. accumulate results from previous step;
    if (warpid > 0) {
      result = OP::apply(shared_memory[warpid-1], result);
    }
    __syncthreads();

    shared_memory[index] = result;
    __syncthreads();
    return OP::remove_volatile(shared_memory[blockDim.x - 1]);
}


template<typename T, CBinaryOperator<T> OP, uint8_t CHUNK = 1>
__global__ void scan(T* src,
                     T* dst,
                     const size_t N,
                     volatile Flag* flags,
                     volatile T* aggregates,
                     volatile T* inclusive_prefixes,
                     uint32_t* counter
                     ){
  // So this kernel is boarderline

  // We are reusing the space as once it's boardcast, we not longer need the space
  extern __shared__ char shared_memory[];
  volatile T* shared_input = (T*)shared_memory;
  volatile FlagVal<T>* shared_flags = (FlagVal<T>*)shared_memory;


  // Registers
  T chunk_registers[CHUNK];
  const uint32_t blockId = get_virtual_block_id(counter);
  __threadfence();

  if(threadIdx.x == 0){
    flags[blockId] = STATUS_INVALID;
  }

  __threadfence();
  __syncthreads();

  {
    const size_t global_offset = blockId * blockDim.x * CHUNK;
    copy_to_shared_memory<T, CHUNK>(shared_input, src, N, OP::identity(), global_offset);
  } // Remove the Global offset register

  {
    T thread_pivot = OP::identity();
    #pragma unroll
    for(uint8_t i = 0; i < CHUNK; i++){
      thread_pivot = OP::apply(thread_pivot, shared_input[threadIdx.x * CHUNK + i]);
      chunk_registers[i] = thread_pivot;
    }

    __syncthreads();
    // Overwrite Shared memory
    shared_input[threadIdx.x] = thread_pivot;
  }
  __syncthreads();

  scan_inclusive_block<T, OP>(shared_input, threadIdx.x);
  // Update chunk registers with local data
  {
    T local_aggregate = threadIdx.x == 0 ? OP::identity() : shared_input[threadIdx.x - 1];
    #pragma unroll
    for(uint8_t i = 0; i < CHUNK; i++){
      chunk_registers[i] = OP::apply(local_aggregate, chunk_registers[i]);
    }
  }
  __syncthreads();


  // Publish the result
  if(threadIdx.x == blockDim.x - 1){
    aggregates[blockId] = shared_input[blockDim.x - 1];
    __threadfence();
    if(blockId != 0){
      flags[blockId] = STATUS_AGGREGATE;
    } else {
      inclusive_prefixes[blockId] = shared_input[blockDim.x - 1];
      __threadfence();
      flags[blockId] = STATUS_PREFIX;
    }
  }
  __threadfence();
  __syncthreads();
  // Do the decoupled for loop
  if(blockId != 0){
    T inclusive_prefix = OP::identity();
    Flag flag = STATUS_INVALID;
    // It is an int because you need that current_index - 1024 can be negative and not overflow
    int32_t current_index = blockId;
    while(flag != STATUS_PREFIX){
      const int thread_index = current_index - (blockDim.x - threadIdx.x);
      FlagVal<T> flag_val;
      flag_val.flag = thread_index < 0 ? STATUS_PREFIX : flags[thread_index];
      if(flag_val.flag == STATUS_PREFIX) {
        flag_val.val = thread_index < 0 ? OP::identity() : inclusive_prefixes[thread_index];
      } else {
        flag_val.val = thread_index < 0 ? OP::identity() : aggregates[thread_index];
      }
      //printf("TID: %d, thread_index: %d\n Flag: %d, Value: %d\n", threadIdx.x + blockDim.x * blockId, thread_index, flag_val.flag, flag_val.val);
      shared_flags[threadIdx.x] = flag_val;

      __syncthreads();

      flag_val = reduce_inclusive_block<FlagVal<T>, FlagOp<T, OP>>(shared_flags, threadIdx.x);

      flag = flag_val.flag;
      if(flag == STATUS_INVALID){
        // Pool again
      } else {
        inclusive_prefix = OP::apply(flag_val.val, inclusive_prefix);
        current_index -= blockDim.x;
      }
    }

    #pragma unroll
    for(uint8_t i = 0; i < CHUNK; i++){
      chunk_registers[i] = OP::apply(inclusive_prefix, chunk_registers[i]);
    }

    if(threadIdx.x == blockDim.x - 1){
      inclusive_prefixes[blockId] = chunk_registers[CHUNK - 1];
      __threadfence();
      flags[blockId] = STATUS_PREFIX;
    }


  }

  __syncthreads();
  #pragma unroll
  for(uint8_t i = 0; i < CHUNK; i++){
    shared_input[threadIdx.x * CHUNK + i] = chunk_registers[i];
  }

  __syncthreads();

  // Copy back to global memory
  {
    const size_t global_offset = blockId * blockDim.x * CHUNK;
    copy_from_shared_memory<T, CHUNK>(dst, shared_input, N, global_offset);
  }
}


#endif