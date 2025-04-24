/**
 * @file label.cuh
 * @author Demiguard (cjen0668@regionh.dk)
 * @brief This is the module for labeling images in GPU
 * @version 0.1
 * @date 2024-12-18
 *
 * @copyright Copyright (c) 2024
 *
 * The underlying algorithm is adopted from:
 * https://github.com/FolkeV/CUDA_CCL at commit @1e9ca96
 */
#pragma once
#include<stdint.h>

#include<iostream>

#include"core/core.cuh"

namespace {
  // "Stolen" function Start
  // ---------- Find the root of a chain ----------
	/**
	 * @brief Finds the root for the label by search
	 *
	 * @param label_chain Memory location where each index is less than the
	 * @param label
	 * @return __device__ uint32_t
	 */
  __device__ uint32_t find_root(const uint32_t* label_chain, uint32_t label) {
	  // Resolve Label
	  uint32_t next = label_chain[label];

	  // Follow chain
	  while(label != next) {
	  	// Move to next
	  	label = next;
	  	next = label_chain[label];
	  }

	  return label;
  }

  // ---------- Label Reduction ----------
  __device__ uint32_t reduction(uint32_t* g_labels, uint32_t label1, uint32_t label2) {
  	// Get next labels
  	uint32_t next1 = (label1 != label2) ? g_labels[label1] : 0;
  	uint32_t next2 = (label1 != label2) ? g_labels[label2] : 0;

  	// Find label1
  	while((label1 != label2) && (label1 != next1)) {
  		// Adopt label
  		label1 = next1;

  		// Fetch next label
  		next1 = g_labels[label1];
  	}

  	// Find label2
  	while((label1 != label2) && (label2 != next2)) {
  		// Adopt label
  		label2 = next2;

  		// Fetch next label
  		next2 = g_labels[label2];
  	}

  	uint32_t label3;
  	// While Labels are different
  	while(label1 != label2) {
  		// Label 2 should be smallest
  		if(label1 < label2) {
				cuda::std::swap(label1, label2);
  		}

  		label3 = atomicMin(&g_labels[label1], label2);
  		label1 = label1 == label3 ? label2 : label3;
  	}

  	return label1 ;
  }

  /**
   * @brief Initializes the labels in g_labels
   *
   * @tparam T
   * @param g_labels pointer to GPU memory holding the labels of size()
   * @param g_image pointer to Image memory
   * @param numCols
   * @param numRows
   * @return __global__
   */
  template<typename T>
  __global__ void init_labels(uint32_t* g_labels, const T* g_image, const size_t num_cols, const size_t num_rows) {
  	// Calculate index
  	const uint32_t ix = blockIdx.x * blockDim.x + threadIdx.x;
  	const uint32_t iy = blockIdx.y * blockDim.y + threadIdx.y;

  	// Check Thread Range
  	if((ix < num_cols) && (iy < num_rows)) {
  		// Fetch five image values
  		const T pyx = g_image[iy*num_cols + ix];

  		// Neighbor Connections
  		const bool nym1x   =  (iy > 0) 					  	 ? (pyx == g_image[(iy-1) * num_cols + ix  ]) : false;
  		const bool nyxm1   =  (ix > 0)  		  			 ? (pyx == g_image[(iy  ) * num_cols + ix-1]) : false;
  		const bool nym1xm1 = ((iy > 0) && (ix > 0)) 		 ? (pyx == g_image[(iy-1) * num_cols + ix-1]) : false;
  		const bool nym1xp1 = ((iy > 0) && (ix < num_cols -1)) ? (pyx == g_image[(iy-1) * num_cols + ix+1]) : false;

  		// Label
  		uint32_t label;

  		// Initialize Label
  		// Label will be chosen in the following order:
  		// NW > N > NE > E > current position
  		label = (nyxm1)   ?  iy   *num_cols + ix-1 : iy*num_cols + ix;
  		label = (nym1xp1) ? (iy-1)*num_cols + ix+1 : label;
  		label = (nym1x)   ? (iy-1)*num_cols + ix   : label;
  		label = (nym1xm1) ? (iy-1)*num_cols + ix-1 : label;

  		// Write to Global Memory
  		g_labels[iy*num_cols + ix] = label;
  	}
  }

  // Resolve Kernel
  __global__ void resolve_labels(uint32_t *g_labels,
  		const size_t numCols, const size_t numRows) {
  	// Calculate index
  	const uint32_t id = ((blockIdx.y * blockDim.y) + threadIdx.y) * numCols +
  							((blockIdx.x * blockDim.x) + threadIdx.x);

  	// Check Thread Range
  	if(id < (numRows* numCols)) {
  		// Resolve Label
  		g_labels[id] = find_root(g_labels, g_labels[id]);
  	}
  }

  // Label Reduction
  template<typename T>
  __global__ void label_reduction(uint32_t *g_labels, const T *g_image,
  		const size_t numCols, const size_t numRows) {
  	// Calculate index
  	const uint32_t iy = ((blockIdx.y * blockDim.y) + threadIdx.y);
  	const uint32_t ix = ((blockIdx.x * blockDim.x) + threadIdx.x);

  	// Check Thread Range
  	if((ix < numCols) && (iy < numRows)) {
  		// Compare Image Values
  		const T pyx = g_image[iy*numCols + ix];
  		const bool nym1x = (iy > 0) ? (pyx == g_image[(iy-1)*numCols + ix]) : false;

  		if(!nym1x) {
  			// Neighbouring values
  			const bool nym1xm1 = ((iy > 0) && (ix >  0)) 		     ? (pyx == g_image[(iy-1) * numCols + ix-1]) : false;
  			const bool nyxm1   =              (ix >  0) 		     ? (pyx == g_image[(iy  ) * numCols + ix-1]) : false;
  			const bool nym1xp1 = ((iy > 0) && (ix < numCols -1)) ? (pyx == g_image[(iy-1) * numCols + ix+1]) : false;

  			if(nym1xp1){
  				// Check Criticals
  				// There are three cases that need a reduction
  				if ((nym1xm1 && nyxm1) || (nym1xm1 && !nyxm1)){
  					// Get labels
  					uint32_t label1 = g_labels[(iy  )*numCols + ix  ];
  					uint32_t label2 = g_labels[(iy-1)*numCols + ix+1];

  					// Reduction
  					reduction(g_labels, label1, label2);
  				}

  				if (!nym1xm1 && nyxm1){
  					// Get labels
  					uint32_t label1 = g_labels[(iy)*numCols + ix  ];
  					uint32_t label2 = g_labels[(iy)*numCols + ix-1];

  					// Reduction
  					reduction(g_labels, label1, label2);
  				}
  			}
  		}
  	}
  }

  // Force background to get label zero;
  template<typename T>
  __global__ void resolve_background(uint32_t *g_labels, const T *g_image,
  		const size_t numCols, const size_t numRows){
  	// Calculate index
  	const uint32_t id = (blockIdx.y * blockDim.y + threadIdx.y) * numCols +
  											blockIdx.x * blockDim.x + threadIdx.x;

  	if(id < numRows*numCols){
  		g_labels[id] = g_image[id] > 0 ? g_labels[id]+1 : 0;
  	}
  }
}
// End of "Stolen" functions

/**
 * @brief Does a GPU connected component labeling on a 2D
 *
 * This functions doesn't do any memory allocations
 *
 * @tparam T - type of the Image
 * @param device_output_labels - pointer to GPU Memory of size
 * @param input_image -
 * @param numCols
 * @param numRows
 * @return dicomNodeError_t
 */
template<typename T>
dicomNodeError_t connectedComponentLabeling2D(
	uint32_t* device_output_labels,
	const T* input_image,
	const size_t num_cols,
	const size_t num_rows
){
	constexpr uint32_t BLOCK_SIZE_X = 32;
	constexpr uint32_t BLOCK_SIZE_Y = 32;

	// Create Grid/Block
	dim3 block{BLOCK_SIZE_X, BLOCK_SIZE_Y, 1};

	const uint32_t grid_x = num_cols % BLOCK_SIZE_X ?
      num_cols / BLOCK_SIZE_X + 1
    : num_cols / BLOCK_SIZE_X;

  const uint32_t grid_y = num_rows % BLOCK_SIZE_Y ?
      num_rows / BLOCK_SIZE_X + 1
    : num_rows / BLOCK_SIZE_X;


	dim3 grid {grid_x, grid_y, 1};

	DicomNodeRunner runner;

	runner
      | [&](){
			init_labels<T><<< grid, block >>>(device_output_labels, input_image, num_cols, num_rows);
			return cudaGetLastError();
		} | [&](){
			resolve_labels <<< grid, block >>>(device_output_labels, num_cols, num_rows);
			return cudaGetLastError();
		} | [&](){
			label_reduction<T><<< grid, block >>>(device_output_labels, input_image, num_cols, num_rows);
			return cudaGetLastError();
		} | [&](){
			resolve_labels <<< grid, block >>>(device_output_labels, num_cols, num_rows);
			return cudaGetLastError();
		} | [&](){
			resolve_background<T><<<grid, block>>>(device_output_labels, input_image, num_cols, num_rows);
			return cudaGetLastError();
		};


	return runner.error();
}

template<typename T>
dicomNodeError_t slicedConnectedComponentLabeling(
  uint32_t* labels,
  const Volume<3, T>& input_volume
){
	const size_t pixels_per_slice = input_volume.extent().x() * input_volume.extent().y();
  size_t offset = 0;
  DicomNodeRunner runner(
		[&](dicomNodeError_t error){
      std::cout << "SlicedConnectedComponentLabeling encountered error: " << error_to_human_readable(error) << "\n"
								<< "At offset: " << offset << "\n"
								<< "At Labels: " << labels + offset << "\n"
								<< "At input data" << input_volume.data << "\n";
		}
	);

  // Yeah I could make a single kernel doing this.
  for(uint32_t i = 0; i < input_volume.extent().z(); i++) {
    runner | [&](){
      return connectedComponentLabeling2D<T>(
        labels + offset,
        input_volume.data + offset,
        input_volume.extent().y(),
        input_volume.extent().x()
      );
    };

    offset += pixels_per_slice;
  }

  return runner.error();
}
