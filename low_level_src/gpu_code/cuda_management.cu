/* This module setup work around CUDA different devices, allowing for better
fitting to the actual GPU device

*/

cudaDeviceProp get_current_device(){
  cudaDeviceProp prop;
  int current_device;
  cudaGetDevice(&current_device);
  cudaGetDeviceProperties(&prop, current_device);
  return prop;
}