PROJECT(OpenPoseWithKinect)

cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++14")

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

list(APPEND CMAKE_MODULE_PATH "${OPENPOSE_DIR}/cmake/Modules")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/Modules")

set(SOURCE_ACQUIRE Acquire_Send_Detect.cpp)
set(EXECUTABLE_ACQUIRE Acquire_Send_Detect)

set(SOURCE_EXTRACT_SKELETON Extract_Skeleton.cpp)
set(EXECUTABLE_EXTRACT_SKELETON Extract_Skeleton)

set(LIBUSB_DIR /usr/include/libusb-1.0/)
set(OPENPOSE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../openpose-1.4.0)
set(CAFFE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../openpose-1.4.0/3rdparty/caffe)
set(freenect2_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../freenect2/lib/cmake/freenect2)
set(OPENCV_3.3.1_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../opencv-3.3.1/build)

set(OPENPOSE_INCLUDE_DIR ${OPENPOSE_DIR}/include)
set(LIBOPENPOSE_DIR ${OPENPOSE_DIR}/build/src/openpose)

set(CAFFE_INCLUDE_DIR ${CAFFE_DIR}/include)

find_package(TensorflowCC COMPONENTS Shared)
find_package(Eigen REQUIRED)
find_package(GFlags REQUIRED)
find_package(Glog REQUIRED)
find_package(freenect2 REQUIRED)
find_package(CUDA REQUIRED)

set(LIBRARIES ${CUDA_LIBS}
${OPENPOSE_DIR}/build/src/openpose/libopenpose.so
${CAFFE_DIR}/build/lib/libcaffe.so
${GFLAGS_LIBRARY}
${GLOG_LIBRARY}
${freenect2_LIBRARIES}
TensorflowCC::Shared
-ljpeg
)

find_package(OpenCV 3.3.1 EXACT PATHS ${OPENCV_3.3.1_PATH} QUIET)
if(OpenCV_FOUND)
  message("-- Found OpenCV at ${OpenCV_DIR}")
  set(LIBRARIES ${LIBRARIES} ${OpenCV_LIBS} -lprotobuf)
endif()

# For desktop Ubuntu 16.04, program fails to run if -lprotobuf is added due to clash, so -lprotobuf is not added here
if(NOT OpenCV_FOUND)
  find_package(OpenCV 3.3.1 EXACT REQUIRED PATHS /opt/ros/kinetic/share/OpenCV-3.3.1-dev/)
  if(OpenCV_FOUND)
    message("-- Found OpenCV at ${OpenCV_DIR}")
    set(LIBRARIES ${LIBRARIES} ${OpenCV_LIBS})
  endif()
endif()

set(PROJECT_INCLUDE_DIRS  ${freenect2_INCLUDE_DIR}
                          ${OpenCV_INCLUDE_DIR}
                          ${FlatBuffers_DIR}
                          ${LIBUSB_DIR}
                          ${OPENPOSE_INCLUDE_DIR}
                          ${CAFFE_INCLUDE_DIR}
                          ${Eigen_INCLUDE_DIRS}
                        )

include_directories(${PROJECT_INCLUDE_DIRS})

if (CUDA_FOUND)
    include_directories(
        ${CUDA_INCLUDE_DIRS}
        ${CAFFE_DIR}/distribute/include
        /usr/local/include/
    )

    add_definitions(-DUSE_CAFFE)

    add_executable(${EXECUTABLE_ACQUIRE} ${SOURCE_ACQUIRE})
    target_link_libraries(${EXECUTABLE_ACQUIRE} ${LIBRARIES}
                          -lnanomsg
                          -lnnxx)

    add_executable(${EXECUTABLE_EXTRACT_SKELETON} ${SOURCE_EXTRACT_SKELETON})
    target_link_libraries(${EXECUTABLE_EXTRACT_SKELETON} ${LIBRARIES}
                          -lnanomsg
                          -lnnxx)
endif ()
