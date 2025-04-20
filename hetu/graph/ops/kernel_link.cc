#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace impl {
bool use_torch_kernel = []() -> bool {
  const char* env_str = std::getenv("HETU_USE_TORCH_KERNEL");
  if (env_str != nullptr) {
    try {
      return std::stoi(env_str) != 0;
    } catch (const std::exception& e) {
      HT_LOG_ERROR << "Invalid HETU_USE_TORCH_KERNEL environment variable value: " << env_str 
                  << ", please provide an integer, default value will be used in this process.";
    }
  }
  return false;
}();

}
}