include(FetchContent)
FetchContent_Declare(
  pybind11
  URL ${CMAKE_SOURCE_DIR}/third_party/pybind11/pybind11-2.6.2.zip
  DOWNLOAD_EXTRACT_TIMESTAMP true
  # GIT_REPOSITORY https://github.com/pybind/pybind11
  # GIT_TAG        v2.6.2
)
FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
  FetchContent_Populate(pybind11)
  add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif()
find_package(Python3 COMPONENTS Interpreter NumPy REQUIRED)

# set(HETU_PY_SRC_DIR ${CMAKE_SOURCE_DIR}/python_refactor)
set(HETU_PY_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Generate operators for Python APIs
set(GEN_OP_YAML ${HETU_PY_SRC_DIR}/hetu/_binding/codegen/ops.yml)
set_directory_properties(PROPERTIES CMAKE_CONFIGURE_DEPENDS ${GEN_OP_YAML})
set(GEN_OP_OUTPUT_DIR ${HETU_PY_SRC_DIR}/hetu/_binding/graph/generated)
set(GEN_OP_STAGING_DIR ${GEN_OP_OUTPUT_DIR}/.staging)

file(MAKE_DIRECTORY ${GEN_OP_OUTPUT_DIR})
file(REMOVE_RECURSE ${GEN_OP_STAGING_DIR})
file(MAKE_DIRECTORY ${GEN_OP_STAGING_DIR})

execute_process(
  COMMAND ${PYTHON_EXECUTABLE} ${HETU_PY_SRC_DIR}/hetu/_binding/codegen/gen_py_ops.py
  --input=${GEN_OP_YAML} --output-dir=${GEN_OP_STAGING_DIR}
)

# Remove outdated files that are no longer needed
file(GLOB GEN_OP_FILES RELATIVE "${GEN_OP_OUTPUT_DIR}/" "${GEN_OP_OUTPUT_DIR}/*")
foreach(FILE ${GEN_OP_FILES})
  if(NOT EXISTS "${GEN_OP_STAGING_DIR}/${FILE}")
    file(REMOVE "${GEN_OP_OUTPUT_DIR}/${FILE}")
  endif()
endforeach()



message("preparing libtorch...")
# set(TORCH_CUDA_ARCH_LIST "8.9" CACHE STRING "CUDA architectures for PyTorch" FORCE)
# # 获取PyTorch信息
# set(Torch_DIR "${CMAKE_SOURCE_DIR}/third_party/libtorch")
# find_package(Torch PATHS ${Torch_DIR} NO_DEFAULT_PATH REQUIRED)
find_package(Torch REQUIRED)
if(TORCH_FOUND)
  # 获取torch真正的位置
  get_target_property(TORCH_ACTUAL_PATH torch INTERFACE_INCLUDE_DIRECTORIES)
  message("libtorch found at: ${TORCH_ACTUAL_PATH}")
  message("TORCH_CXX_FLAGS: ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
  target_link_libraries(hetu_C PUBLIC ${TORCH_LIBRARIES})
  target_include_directories(hetu_C PUBLIC ${TORCH_INCLUDE_DIRS})
  target_compile_definitions(hetu_C PUBLIC -DTORCH_AVAILABLE)
endif(TORCH_FOUND)
message("preparing libtorch done")


# Override outdated files that have been modified
file(GLOB GEN_OP_FILES RELATIVE "${GEN_OP_STAGING_DIR}/" "${GEN_OP_STAGING_DIR}/*")
foreach(FILE ${GEN_OP_FILES})
  execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different "${GEN_OP_STAGING_DIR}/${FILE}" "${GEN_OP_OUTPUT_DIR}")
endforeach()
file(REMOVE_RECURSE ${GEN_OP_STAGING_DIR})

file(GLOB_RECURSE HETU_PY_SRC ${HETU_PY_SRC_DIR}/hetu/*.cc)
# TODO: remove the entire directories
list(FILTER HETU_PY_SRC EXCLUDE REGEX "${HETU_PY_SRC_DIR}/hetu/_binding/autograd/*")
list(FILTER HETU_PY_SRC EXCLUDE REGEX "${HETU_PY_SRC_DIR}/hetu/_binding/execution/*")
pybind11_add_module(_hetu_core ${HETU_PY_SRC})
target_include_directories(_hetu_core PUBLIC ${Python3_INCLUDE_DIRS})
target_link_libraries(_hetu_core PUBLIC ${Python3_LIBRARIES})
target_link_libraries(_hetu_core PUBLIC Python3::NumPy)
target_link_libraries(_hetu_core PUBLIC hetu_C)
target_include_directories(_hetu_core PUBLIC ${HETU_PY_SRC_DIR})
