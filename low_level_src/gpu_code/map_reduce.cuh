#pragma once

#include<stdint.h>
#include<concepts>
#include<assert.h>

#include"core/core.cuh"

#include"map_reduce.cuh"


namespace { // Private function

enum STATUS_FLAGS: uint8_t {
  STATUS_INVALID=0,
  STATUS_AGGREGATE=1,
  STATUS_PREFIX=2,
};

using Flag = STATUS_FLAGS;

template<typename T, CBinaryOperator<T> OP>
struct FlagVal {
    T val;
    Flag flag;

    __host__ __device__ inline FlagVal() : val(), flag(STATUS_INVALID) {}
    __host__ __device__ inline FlagVal(const Flag& _flag, const T& value) {
      val = value;
      flag = _flag;
    }

    __host__ __device__ inline FlagVal(volatile FlagVal& flag_val) {
      val = OP::remove_volatile(flag_val.val);
      flag = flag_val.flag;
    }

    __host__ __device__ inline void operator=(const FlagVal& flag_val) volatile {
      val = flag_val.val;
      flag = flag_val.flag;
    }
};

__device__ uint32_t get_shared_id(uint32_t* counter){
  __shared__ uint32_t address;
  if(threadIdx.x == 0){
    uint32_t local = atomicAdd(counter, 1);
    address = local;
  }
  __syncthreads();
  return address;
}

// Sadly there is no way to mark this as static
template<typename T, CBinaryOperator<T> OP>
class FlagOp {
  public:
    static __host__ __device__ inline FlagVal<T, OP> identity() {return FlagVal<T, OP>(STATUS_AGGREGATE, OP::identity());}
    static __host__ __device__ inline bool equals(const FlagVal<T, OP> a, const FlagVal<T, OP> b){
      return OP::equals(a.val, b.val) && a.flag == b.flag;
    }

    static __host__ __device__ inline FlagVal<T, OP> map_to(T in, int64_t index, Flag flag) {
      return FlagVal<T, OP>(flag, in);
    }

    static __host__ __device__ inline FlagVal<T, OP> apply(const FlagVal<T, OP> a, const FlagVal<T, OP> b){
      // This is here for return value optimization
      FlagVal<T, OP> returnStruct;
      if(b.flag == STATUS_PREFIX){
        returnStruct.flag = STATUS_PREFIX;
        returnStruct.val = b.val;
      } else if(a.flag == STATUS_INVALID || b.flag == STATUS_INVALID){
        returnStruct.flag = STATUS_INVALID;
      } else if (a.flag == STATUS_PREFIX) { // b.flag == STATUS_AGGREGATE
        returnStruct.flag = STATUS_PREFIX;
        returnStruct.val = OP::apply(
          a.val, b.val);
      } else { // (a,b).flag == STATUS_AGGREGATE
        returnStruct.flag = STATUS_AGGREGATE;
        returnStruct.val = OP::apply(a.val, b.val);
      }
      return returnStruct;
    }

    static __host__ __device__ inline FlagVal<T, OP> remove_volatile(volatile FlagVal<T, OP>& flag_val){
      FlagVal<T, OP> fv = flag_val;
      return fv;
    }
};


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
    dst_shared[local_index] = remove_volatile(element);
  }

  __syncthreads();
}

/**
 * @brief
 *
 * @tparam T The type of the binary operator
 * @tparam OP The binary operator in the scan
 * @param data Data array
 * @param index
 * @return __device__
 */
template<typename T, CBinaryOperator<T> OP>
__device__ inline T scan_inclusive_warp_kogge_stone(volatile T* data, const size_t index){
  const uint8_t lane = index & (WARP_SIZE - 1);

  #pragma unroll
  for(uint8_t i = 1; i < WARP_SIZE; i <<= 1){
    if(i <= lane){
      data[index] = OP::apply(
        OP::remove_volatile(data[index - i]),
        OP::remove_volatile(data[index]));
    }
    __syncwarp();
  }

  return OP::remove_volatile(data[index]);
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
      result = OP::apply(
        OP::remove_volatile(shared_memory[warpid-1]),
        result
      );
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
    T res = shared_memory[blockDim.x - 1];
    return res;
}

template<uint8_t CHUNK, typename OP, typename T_IN, typename T_OUT,  typename... Args>
  requires MappingBinaryOperator<OP, T_IN, T_OUT, Args...>
__global__ void scan_kernel(T_IN* src,
                             T_OUT* dst,
                             const size_t N,
                             volatile Flag* flags,
                             volatile T_OUT* aggregates,
                             volatile T_OUT* inclusive_prefixes,
                             uint32_t* counter,
                             Args... args
                           ){
  // So this kernel is boarderline

  // We are reusing the space as once it's boardcast, we not longer need the space
  extern __shared__ char shared_memory[];
  volatile T_OUT* shared_input = (T_OUT*)shared_memory;
  volatile FlagVal<T_OUT, OP>* shared_flags = (FlagVal<T_OUT, OP>*)shared_memory;


  // Registers
  T_OUT chunk_registers[CHUNK];
  const uint32_t blockId = get_shared_id(counter);
  __threadfence();

  if(threadIdx.x == 0){
    flags[blockId] = STATUS_INVALID;
    aggregates[blockId] = OP::identity();
    inclusive_prefixes[blockId] = OP::identity();
  }

  __threadfence();
  __syncthreads();

  {
    const size_t global_offset = blockId * blockDim.x * CHUNK;

    map_into_shared_memory<CHUNK, OP,T_IN, T_OUT, Args...>(shared_input, src, N, global_offset, args...);
  } // Remove the Global offset register

  {
    T_OUT thread_pivot = OP::identity();
    #pragma unroll
    for(uint8_t i = 0; i < CHUNK; i++){
      thread_pivot = OP::apply(thread_pivot, OP::remove_volatile(shared_input[threadIdx.x * CHUNK + i]));
      chunk_registers[i] = thread_pivot;
    }

    __syncthreads();
    // Overwrite Shared memory
    shared_input[threadIdx.x] = thread_pivot;
  }
  __syncthreads();

  scan_inclusive_block<T_OUT, OP>(shared_input, threadIdx.x);
  // Update chunk registers with local data
  {
    T_OUT local_aggregate = threadIdx.x == 0 ? OP::identity() : OP::remove_volatile(shared_input[threadIdx.x - 1]);
    #pragma unroll
    for(uint8_t i = 0; i < CHUNK; i++){
      chunk_registers[i] = OP::apply(local_aggregate, chunk_registers[i]);
    }
  }
  __syncthreads();

  // Publish the result
  if(threadIdx.x == blockDim.x - 1){
    aggregates[blockId] = OP::remove_volatile(shared_input[blockDim.x - 1]);
    __threadfence();
    if(blockId != 0){
      flags[blockId] = STATUS_AGGREGATE;
    } else {
      inclusive_prefixes[blockId] = OP::remove_volatile(shared_input[blockDim.x - 1]);
      __threadfence();
      flags[blockId] = STATUS_PREFIX;
    }
  }
  __threadfence();
  __syncthreads();
  // Do the decoupled for loop
  if(blockId != 0){
    T_OUT inclusive_prefix = OP::identity();
    Flag flag = STATUS_INVALID;
    // It is an int because you need that current_index - 1024 can be negative and not overflow
    int32_t current_index = blockId;
    while(flag != STATUS_PREFIX){
      __syncthreads();
      const int thread_index = current_index - (blockDim.x - threadIdx.x);
      shared_flags[threadIdx.x].flag = thread_index < 0 ? STATUS_PREFIX : flags[thread_index];
      if(shared_flags[threadIdx.x].flag == STATUS_PREFIX) {
        shared_flags[threadIdx.x].val = thread_index < 0 ? OP::identity() : OP::remove_volatile(inclusive_prefixes[thread_index]);
      } else {
        shared_flags[threadIdx.x].val = thread_index < 0 ? OP::identity() : OP::remove_volatile(aggregates[thread_index]);
      }

      __syncthreads();

      const FlagVal flag_val = reduce_inclusive_block<FlagVal<T_OUT, OP>, FlagOp<T_OUT, OP>>(shared_flags, threadIdx.x);

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
    copy_from_shared_memory<T_OUT, CHUNK>(dst, shared_input, N, global_offset);
  }
}

template<uint8_t CHUNK, typename OP, typename T_IN, typename T_OUT,  typename... Args>
  requires MappingBinaryOperator<OP, T_IN, T_OUT, Args...>
__global__ void reduce_kernel(T_IN* src,
                       T_OUT* dst,
                       const size_t N,
                       volatile Flag* flags,
                       volatile T_OUT* aggregates,
                       volatile T_OUT* inclusive_prefixes,
                       uint32_t* counter,
                       Args... args
                     ){
  // So this kernel is boarderline

  // We are reusing the space as once it's boardcast, we not longer need the space
  extern __shared__ char shared_memory[];
  volatile T_OUT* shared_input = (T_OUT*)shared_memory;
  volatile FlagVal<T_OUT, OP>* shared_flags = (FlagVal<T_OUT, OP>*)shared_memory;


  // Registers
  T_OUT chunk_registers[CHUNK];
  const uint32_t blockId = get_shared_id(counter);
  __threadfence();

  if(threadIdx.x == 0){
    flags[blockId] = STATUS_INVALID;
    aggregates[blockId] = OP::identity();
    inclusive_prefixes[blockId] = OP::identity();
  }

  __threadfence();
  __syncthreads();

  {
    const size_t global_offset = blockId * blockDim.x * CHUNK;

    map_into_shared_memory<CHUNK, OP,T_IN, T_OUT, Args...>(shared_input, src, N, global_offset, args...);
  } // Remove the Global offset register

  {
    T_OUT thread_pivot = OP::identity();
    #pragma unroll
    for(uint8_t i = 0; i < CHUNK; i++){
      thread_pivot = OP::apply(thread_pivot, OP::remove_volatile(shared_input[threadIdx.x * CHUNK + i]));
      chunk_registers[i] = thread_pivot;
    }

    __syncthreads();
    // Overwrite Shared memory
    shared_input[threadIdx.x] = thread_pivot;
  }
  __syncthreads();

  scan_inclusive_block<T_OUT, OP>(shared_input, threadIdx.x);
  // Update chunk registers with local data
  {
    T_OUT local_aggregate = threadIdx.x == 0 ? OP::identity() : OP::remove_volatile(shared_input[threadIdx.x - 1]);
    #pragma unroll
    for(uint8_t i = 0; i < CHUNK; i++){
      chunk_registers[i] = OP::apply(local_aggregate, chunk_registers[i]);
    }
  }
  __syncthreads();


  // Publish the result
  if(threadIdx.x == blockDim.x - 1){
    aggregates[blockId] = OP::remove_volatile(shared_input[blockDim.x - 1]);
    __threadfence();
    if(blockId != 0){
      flags[blockId] = STATUS_AGGREGATE;
    } else {
      inclusive_prefixes[blockId] = OP::remove_volatile(shared_input[blockDim.x - 1]);
      __threadfence();
      flags[blockId] = STATUS_PREFIX;
    }
  }
  __threadfence();
  __syncthreads();
  // Do the decoupled for loop
  if(blockId != 0){
    T_OUT inclusive_prefix = OP::identity();
    Flag flag = STATUS_INVALID;
    // It is an int because you need that current_index - 1024 can be negative and not overflow
    int32_t current_index = blockId;
    while(flag != STATUS_PREFIX){
      __syncthreads();
      const int thread_index = current_index - (blockDim.x - threadIdx.x);
      shared_flags[threadIdx.x].flag = thread_index < 0 ? STATUS_PREFIX : flags[thread_index];
      if(shared_flags[threadIdx.x].flag == STATUS_PREFIX) {
        shared_flags[threadIdx.x].val = thread_index < 0 ? OP::identity() : OP::remove_volatile(inclusive_prefixes[thread_index]);
      } else {
        shared_flags[threadIdx.x].val = thread_index < 0 ? OP::identity() : OP::remove_volatile(aggregates[thread_index]);
      }

      __syncthreads();

      const FlagVal flag_val = reduce_inclusive_block<FlagVal<T_OUT, OP>, FlagOp<T_OUT, OP>>(shared_flags, threadIdx.x);

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
  if(blockId == gridDim.x - 1 && threadIdx.x == blockDim.x - 1){
      dst[0] = chunk_registers[CHUNK - 1];
  }
}

} // End of anonymous namespace


/**
 * @brief performs a scan on the data array
 *
 * @tparam chunk Number of elements each thread processes
 * @tparam OP
 * @tparam T_IN
 * @tparam T_OUT
 * @tparam Args
 * @param data host array of length data_size and of type T_IN
 * @param data_size length of data and out
 * @param output host array of length data_size and of type T_OUT
 * @param args argumentment pack such that OP::map_to(data[n], args...) = t_out[n]
 * @return requires
 */
template<uint8_t chunk, typename OP, typename T_IN, typename T_OUT, typename... Args>
  requires MappingBinaryOperator<OP, T_IN, T_OUT, Args...>
cudaError_t scan(const T_IN* data, const size_t data_size, T_OUT* output, const Args... args){
  assert(data != nullptr);
  assert(output != nullptr);
  assert(data_size != 0);

  T_IN*     device_in = nullptr;
  T_OUT*    device_out = nullptr;
  Flag*     device_flags = nullptr;
  T_OUT*    device_aggregates = nullptr;
  T_OUT*    device_prefixes = nullptr;
  uint32_t* device_counter = nullptr;


  constexpr size_t shared_memory_size = max(chunk * sizeof(T_OUT) * SCAN_BLOCK_SIZE, SCAN_BLOCK_SIZE * sizeof(FlagVal<T_OUT, OP>));
  const dim3 grid = get_grid<chunk>(data_size, SCAN_BLOCK_SIZE);


  auto error_function = [&](cudaError_t error){
    free_device_memory(device_in, device_out, device_flags, device_aggregates,
                       device_prefixes, device_counter);
  };

  CudaRunner runner{error_function};

  runner
    | [&](){return cudaMalloc(&device_in, sizeof(T_IN) * data_size);}
    | [&](){return cudaMalloc(&device_out, sizeof(T_OUT) * data_size);}
    | [&](){return cudaMalloc(&device_flags, sizeof(Flag) * grid.x);}
    | [&](){return cudaMalloc(&device_aggregates, sizeof(T_OUT) * grid.x);}
    | [&](){return cudaMalloc(&device_prefixes, sizeof(T_OUT) * grid.x);}
    | [&](){return cudaMalloc(&device_counter, sizeof(uint32_t));}
    | [&](){return cudaMemcpy(device_in, data, sizeof(T_IN) * data_size, cudaMemcpyHostToDevice);}
    | [&](){return cudaMemset(device_counter, 0, sizeof(uint32_t));}
    | [&](){scan_kernel<chunk, OP, T_IN, T_OUT, Args...>
              <<<grid, SCAN_BLOCK_SIZE, shared_memory_size>>>(
                device_in, device_out, data_size, device_flags, device_aggregates, device_prefixes, device_counter, args...
              );
            return cudaGetLastError();
           }
    | [&](){return cudaMemcpy(output, device_out, sizeof(T_OUT) * data_size, cudaMemcpyDeviceToHost);}
    | [&](){free_device_memory(device_in, device_out, device_flags,
                                device_aggregates, device_prefixes,
                                device_counter);
            return cudaSuccess;};

  return runner.error();
}

/**
 * @brief Performs a reduce at device level
 *
 * @tparam CHUNK the number of element each thread
 * @tparam OP
 * @tparam T_IN
 * @tparam T_OUT
 * @tparam Args
 * @param data
 * @param data_size
 * @param output
 * @param args
 * @return requires
 */
template<uint8_t chunk, typename OP, typename T_IN, typename T_OUT, typename... Args>
  requires MappingBinaryOperator<OP, T_IN, T_OUT, Args...>
cudaError_t reduce(const T_IN* data, const size_t data_size, T_OUT* output, const Args... args){
  T_IN*     device_in = nullptr;
  T_OUT*    device_out = nullptr;
  Flag*     device_flags = nullptr;
  T_OUT*    device_aggregates = nullptr;
  T_OUT*    device_prefixes = nullptr;
  uint32_t* device_counter = nullptr;

  const size_t shared_memory_size = max(chunk * sizeof(T_OUT) * SCAN_BLOCK_SIZE, SCAN_BLOCK_SIZE * sizeof(FlagVal<T_OUT, OP>));
  const dim3 grid = get_grid<chunk>(data_size, SCAN_BLOCK_SIZE);

  auto error_function = [&](cudaError_t error){
    free_device_memory(&device_in, &device_out, &device_flags, &device_aggregates,
                       &device_prefixes, &device_counter);
  };

  CudaRunner runner{error_function};

  runner
    | [&](){ return cudaMalloc(&device_in, sizeof(T_IN) * data_size);}
    | [&](){ return cudaMalloc(&device_out, sizeof(T_OUT));}
    | [&](){ return cudaMalloc(&device_flags, sizeof(Flag) * grid.x);}
    | [&](){ return cudaMalloc(&device_aggregates, sizeof(T_OUT) * grid.x);}
    | [&](){ return cudaMalloc(&device_prefixes, sizeof(T_OUT) * grid.x);}
    | [&](){ return cudaMalloc(&device_counter, sizeof(uint32_t));}
    | [&](){ return cudaMemcpy(device_in, data, sizeof(T_IN) * data_size, cudaMemcpyDefault);}
    | [&](){ return cudaMemset(device_counter, 0, sizeof(uint32_t));}
    | [&](){
      reduce_kernel<chunk, OP, T_IN, T_OUT, Args...>
              <<<grid, SCAN_BLOCK_SIZE, shared_memory_size>>>(
                device_in, device_out, data_size, device_flags, device_aggregates, device_prefixes, device_counter, args...
              );
            return cudaGetLastError();
           }
    | [&](){ return cudaMemcpy(output, device_out, sizeof(T_OUT), cudaMemcpyDefault);}
    | [&](){ free_device_memory(&device_in, &device_out, &device_flags,
                                &device_aggregates, &device_prefixes,
                                &device_counter);
            return cudaSuccess;};

  return runner.error();
}
