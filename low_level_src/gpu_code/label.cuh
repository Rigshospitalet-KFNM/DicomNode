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
#include"core/core.cuh"

namespace {
  // "Stolen" function Start
  // ---------- Find the root of a chain ----------
  __device__ uint32_t find_root(uint32_t* labels, uint32_t label) {
	  // Resolve Label
	  uint32_t next = labels[label];

	  // Follow chain
	  while(label != next) {
	  	// Move to next
	  	label = next;
	  	next = labels[label];
	  }

	  // Return label
	  return(label);
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
  			// Swap Labels
  			label1 = label1 ^ label2;
  			label2 = label1 ^ label2;
  			label1 = label1 ^ label2;
  		}

  		// AtomicMin label1 to label2
  		label3 = atomicMin(&g_labels[label1], label2);
  		label1 = (label1 == label3) ? label2 : label3;
  	}

  	// Return label1
  	return(label1);
  }

  template<typename T>
  __global__ void init_labels(uint32_t* g_labels, const T* g_image, const size_t numCols, const size_t numRows) {
  	// Calculate index
  	const uint32_t ix = (blockIdx.x * blockDim.x) + threadIdx.x;
  	const uint32_t iy = (blockIdx.y * blockDim.y) + threadIdx.y;

  	// Check Thread Range
  	if((ix < numCols) && (iy < numRows)) {
  		// Fetch five image values
  		const unsigned char pyx = g_image[iy*numCols + ix];

  		// Neighbour Connections
  		const bool nym1x   =  (iy > 0) 					  	 ? (pyx == g_image[(iy-1) * numCols + ix  ]) : false;
  		const bool nyxm1   =  (ix > 0)  		  			 ? (pyx == g_image[(iy  ) * numCols + ix-1]) : false;
  		const bool nym1xm1 = ((iy > 0) && (ix > 0)) 		 ? (pyx == g_image[(iy-1) * numCols + ix-1]) : false;
  		const bool nym1xp1 = ((iy > 0) && (ix < numCols -1)) ? (pyx == g_image[(iy-1) * numCols + ix+1]) : false;

  		// Label
  		uint32_t label;

  		// Initialise Label
  		// Label will be chosen in the following order:
  		// NW > N > NE > E > current position
  		label = (nyxm1)   ?  iy   *numCols + ix-1 : iy*numCols + ix;
  		label = (nym1xp1) ? (iy-1)*numCols + ix+1 : label;
  		label = (nym1x)   ? (iy-1)*numCols + ix   : label;
  		label = (nym1xm1) ? (iy-1)*numCols + ix-1 : label;

  		// Write to Global Memory
  		g_labels[iy*numCols + ix] = label;
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
  	const uint32_t id = ((blockIdx.y * blockDim.y) + threadIdx.y) * numCols +
  							((blockIdx.x * blockDim.x) + threadIdx.x);

  	if(id < numRows*numCols){
  		g_labels[id] = (g_image[id] > 0) ? g_labels[id]+1 : 0;
  	}
  }
}

template<typename T>
dicomNodeError_t connectedComponentLabeling2D(
	uint32_t* device_output_labels,
	T* inputImg,
	size_t numCols,
	size_t numRows
){
	// I should do some testing with different sizes but ooh well
	constexpr uint32_t BLOCK_SIZE_X = 32;
	constexpr uint32_t BLOCK_SIZE_Y = 4;

	// Create Grid/Block
	dim3 block (BLOCK_SIZE_X, BLOCK_SIZE_Y);
	dim3 grid ((numCols+BLOCK_SIZE_X-1)/BLOCK_SIZE_X,
			(numRows+BLOCK_SIZE_Y-1)/BLOCK_SIZE_Y);

	DicomNodeRunner runner;

	runner
		|   [&](){
			// Initialise labels
			init_labels<<< grid, block >>>(device_output_labels, inputImg, numCols, numRows);
			return cudaGetLastError();
		} | [&](){
			// Analysis
			resolve_labels <<< grid, block >>>(device_output_labels, numCols, numRows);
			return cudaGetLastError();
		} | [&](){
			// Label Reduction
			label_reduction <<< grid, block >>>(device_output_labels, inputImg, numCols, numRows);
			return cudaGetLastError();
		} | [&](){
			// Analysis
			resolve_labels <<< grid, block >>>(device_output_labels, numCols, numRows);
			return cudaGetLastError();
		} | [&](){
			// Force background to have label zero;
			resolve_background<<<grid, block>>>(device_output_labels, inputImg, numCols, numRows);
			return cudaGetLastError();
		};


	return runner.error();
}
// End of "Stolen" functions
