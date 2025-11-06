cmake_minimum_required(VERSION 3.20)
project(mlir-hello-world)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MLIR
find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)

# Include directories
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

# Define the executable
add_executable(mlir-hello-world main.cpp)

# Link against MLIR libraries
target_link_libraries(mlir-hello-world
  PRIVATE
    MLIRIR
    MLIRFunc
    MLIRFuncDialect
    MLIRArithDialect
    MLIRSupport
    MLIRParser
)
