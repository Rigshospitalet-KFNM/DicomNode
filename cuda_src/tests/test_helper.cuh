#pragma once

template<typename T>
class DeviceArray {
  T* device_ptr = nullptr;
  size_t elements;

public:
  explicit DeviceArray(size_t count) noexcept  : elements(count) {
    cudaMalloc(&device_ptr, sizeof(T) * elements);
  }

  ~DeviceArray() {
    cudaFree(device_ptr);
  }

  T* get(){ return device_ptr; }
  size_t size() { return elements * sizeof(T); }

  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;
  DeviceArray(DeviceArray&& other): device_ptr(other.device_ptr), elements(other.elements) {
    other.device_ptr = nullptr;
    other.elements = 0;
  }

  DeviceArray& operator=(DeviceArray&& other) {
    if (device_ptr) {
      cudaFree(device_ptr);
    }
    device_ptr = other.device_ptr;
    elements = other.elements;

    other.device_ptr = nullptr;
    other.elements = 0;
  }
};