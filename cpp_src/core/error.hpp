#pragma once
#include<stdint.h>

#include<functional>

#include"declarations.hpp"

enum dicomNodeError_t: u32 {
  SUCCESS = 0,
  NOT_LINEAR_INDEPENDENT = 1,
  INPUT_TYPE_ERROR = 2,
  NON_POSITIVE_SHAPE = 3,
  POINTER_IS_NOT_A_DEVICE_PTR = 4,
  UNABLE_TO_ACQUIRE_BUFFER = 5,
  INPUT_SIZE_MISMATCH = 6,
  ALREADY_ALLOCATED_OBJECT = 7, // This is triggered if you try to allocate to an object that's already alocated
};

class DicomNodeRunner {
  public:
    DicomNodeRunner() : m_error_function([](dicomNodeError_t _){}){}

    DicomNodeRunner(std::function<void(dicomNodeError_t)> error_funciton)
    : m_error_function(error_funciton) {}

  DicomNodeRunner& operator|(std::function<dicomNodeError_t()> func){
    if(m_error == dicomNodeError_t::SUCCESS){
      m_error = func();
      if(m_error != dicomNodeError_t::SUCCESS){
        m_error_function(m_error);
      }
    }
    return *this;
  }

  template<typename F>
    requires std::invocable<F> &&
             std::same_as<std::invoke_result_t<F>, dicomNodeError_t>
  DicomNodeRunner& operator|(F&& func){
    if(m_error == dicomNodeError_t::SUCCESS){
      m_error = func();
      if(m_error != dicomNodeError_t::SUCCESS){
        m_error_function(m_error);
      }
    }
    return *this;
  }

  dicomNodeError_t error() const {
    return m_error;
  }

  private:
    std::function<void(dicomNodeError_t)> m_error_function;
    dicomNodeError_t m_error = dicomNodeError_t::SUCCESS;
};