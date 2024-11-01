#include"utilities.cuh"

std::string get_byte_string (size_t bytes){
  std::stringstream ss;
  if(bytes < 1024){
    ss << bytes << " B";
    return ss.str();
  }
  bytes >>= 10;
  if(bytes < 1024){
    ss << bytes << " kB";
    return ss.str();
  }
  bytes >>= 10;
  if(bytes < 1024){
    ss << bytes << " MB";
    return ss.str();
  }

  bytes >>= 10;
  ss << bytes << " GB";
  return ss.str();
}
