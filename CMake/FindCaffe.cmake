#/*============================================================================
#
#  NifTK: A software platform for medical image computing.
#
#  Copyright (c) University College London (UCL). All rights reserved.
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.
#
#  See LICENSE.txt in the top level directory for details.
#
#============================================================================*/

set(Caffe_FOUND)

set(Caffe_DIR @Caffe_DIR@ CACHE PATH "Directory containing Caffe installation")

message(STATUS "Using custom Find${CMAKE_FIND_PACKAGE_NAME} module")

#set(Boost_USE_STATIC_LIBS ON)
#set(Boost_USE_MULTITHREAD ON)
#set(Boost_USE_STATIC_RUNTIME OFF)

#add_definitions(-DBOOST_ALL_NO_LIB)

find_package(Boost COMPONENTS system thread filesystem date_time python regex)
find_package(GFlags REQUIRED)
find_package(GLog REQUIRED)

#set(PROTOBUF_SRC_ROOT_FOLDER ${CMAKE_INSTALL_PREFIX})

find_package(ProtoBuf REQUIRED)
find_package(HDF5 REQUIRED)
#find_package(LMDB REQUIRED)
#find_package(LevelDB REQUIRED)
#find_package(Snappy REQUIRED)
#find_package(OpenCV REQUIRED)
find_package(OpenBLAS REQUIRED)

#find_package(${CMAKE_FIND_PACKAGE_NAME} CONFIG NO_CMAKE_PACKAGE_REGISTRY NO_CMAKE_SYSTEM_PACKAGE_REGISTRY)

#set(CAFFE_FOUND ${${CMAKE_FIND_PACKAGE_NAME}_FOUND})	

set(Caffe_INCLUDE_DIR
  NAME caffe.hpp
  PATHS ${Caffe_DIR}/include ${Caffe_DIR}/include/caffe
  NO_DEFAULT_PATH
)

set(Caffe_LIBRARY_DIR ${Caffe_DIR}/lib)

set(Caffe_LIBRARY )

if(${CMAKE_BUILD_TYPE} STREQUAL "Release")

  find_library(Caffe_LIBRARY NAMES caffe
               PATHS ${Caffe_LIBRARY_DIR}
               PATH_SUFFIXES Release
               NO_DEFAULT_PATH)

elseif(${CMAKE_BUILD_TYPE} STREQUAL "Debug")

  find_library(Caffe_LIBRARY NAMES caffed
               PATHS ${Caffe_LIBRARY_DIR}
               PATH_SUFFIXES Debug
               NO_DEFAULT_PATH)

endif()

if(Caffe_LIBRARY AND Caffe_INCLUDE_DIR)

  set(Caffe_FOUND 1)

endif()

message( "Caffe_INCLUDE_DIR: ${Caffe_INCLUDE_DIR}" )
message( "Caffe_LIBRARY_DIR: ${Caffe_LIBRARY_DIR}" )
message( "Caffe_LIBRARY:     ${Caffe_LIBRARY}" )
