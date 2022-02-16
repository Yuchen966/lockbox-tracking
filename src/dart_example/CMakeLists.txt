project("dart_example")
cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 11)

if( NOT CMAKE_BUILD_TYPE )
   set( CMAKE_BUILD_TYPE Release )
endif()

add_definitions(-std=c++11)

find_path(DART_ROOT NAMES src/tracker.h PATHS ${PROJECT_SOURCE_DIR}/../dart)
message(STATUS "Found dart is at ${DART_ROOT}")
include_directories(${DART_ROOT}/src)
link_directories(${DART_ROOT}/lib)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${DART_ROOT}/CMakeModules/")

# find packages with modules
find_package(Pangolin REQUIRED PATHS ${PROJECT_SOURCE_DIR}/**)
find_package(DEPTHSENSE)
find_package(CUDA REQUIRED)

# find packages with pkg-config
find_package(PkgConfig)
pkg_check_modules(eigen3 REQUIRED eigen3)
pkg_check_modules(matheval REQUIRED libmatheval)

if(DEPTHSENSE_FOUND)
    add_definitions(-DHAS_INTEL)
    include_directories(${DEPTHSENSE_INCLUDE_DIR})
    link_libraries(${DEPTHSENSE_LIBRARIES})
else()
    message(STATUS "no DS")
endif()

# find packages of catkin and opencv
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  cv_bridge
  sensor_msgs
  image_transport
)
find_package(OpenCV REQUIRED)

# include directories
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

include_directories(  ${Pangolin_INCLUDE_DIRS} )
link_directories(     ${Pangolin_LIBRARY_DIRS} )
link_libraries(       ${Pangolin_LIBRARIES}    )
message("${Pangolin_INCLUDE_DIRS}, ${Pangolin_LIBRARY_DIRS}, ${Pangolin_LIBRARIES}")

include_directories(
    ${eigen3_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src
)

catkin_package(
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS roscpp rospy std_msgs sensor_msgs message_runtime image_transport cv_bridge
  DEPENDS OpenCV
)

##########################################################################
#   Build GPU or CPU-only executable depending on cuda                   #
##########################################################################

include_directories( ${CUDA_INCLUDE_DIRS} )
cuda_include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# set(CUDA_NVCC_FLAGS "-arch=sm_35" "--use_fast_math"  "-O3" "--ptxas-options=--verbose") # "-fmad=false" "-DTHRUST_DEVICE_BACKEND=THRUST_DEVICE_BACKEND_OMP"
set(CUDA_NVCC_FLAGS "-arch=sm_86" "--use_fast_math"  "-O3" "--ptxas-options=--verbose") # "-fmad=false" "-DTHRUST_DEVICE_BACKEND=THRUST_DEVICE_BACKEND_OMP"
add_definitions(-DCUDA_BUILD)

set(all_sources ${sources} ${cpu_sources} ${gpu_sources})
link_directories( ${CUDA_TOOLKIT_ROOT_DIR}/lib64/)

cuda_add_executable(
    #executable
    track_space_justin_demo
    #sources
    ${all_sources}
    #main
    node/track_space_justin_demo.cpp
)
target_link_libraries(track_space_justin_demo dart)

## Link libraries
find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_DIR})
target_link_libraries(track_space_justin_demo ${PNG_LIBRARY})

find_package(JPEG REQUIRED)
include_directories(${JPEG_INCLUDE_DIR})
target_link_libraries(track_space_justin_demo ${JPEG_LIBRARY})

cuda_add_executable(
    #executable
    track_lockbox_mov
    #sources
    ${all_sources}
    #main
    node/track_lockbox_mov.cpp
)
target_link_libraries(track_lockbox_mov dart ${PNG_LIBRARY} ${JPEG_LIBRARY})

cuda_add_executable(
    #executable
    track_lockbox_rigid
    #sources
    ${all_sources}
    #main
    node/track_lockbox_rigid.cpp
)
target_link_libraries(track_lockbox_rigid dart ${PNG_LIBRARY} ${JPEG_LIBRARY})

cuda_add_executable(
    #executable
    track_lockbox_rigid_exp
    #sources
    ${all_sources}
    #main
    node/track_lockbox_rigid_exp.cpp
)
target_link_libraries(track_lockbox_rigid_exp dart ${PNG_LIBRARY} ${JPEG_LIBRARY})

cuda_add_executable(
    #executable
    track_lockbox_mov_exp
    #sources
    ${all_sources}
    #main
    node/track_lockbox_mov_exp.cpp
)
target_link_libraries(track_lockbox_mov_exp dart ${PNG_LIBRARY} ${JPEG_LIBRARY})

