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


#-----------------------------------------------------------------------------
# Caffe
#-----------------------------------------------------------------------------

# Sanity checks
if(DEFINED Caffe_DIR AND NOT EXISTS ${Caffe_DIR})
  message(FATAL_ERROR "Caffe_DIR variable is defined but corresponds to non-existing directory \"${Caffe_ROOT}\".")
endif()

if(MITK_USE_Caffe)

  # Microsoft Caffe Windows branch, forked from https://github.com/BVLC/caffe,
  # with Eli's mods (SPIE 2017) merged in, and fixes to get working on Windows, Linux and Mac.
  set(version "7c758f5630")

  set(location "${NIFTK_EP_TARBALL_LOCATION}/caffe-${version}.tar.gz")

  niftkMacroDefineExternalProjectVariables(Caffe ${version} ${location})

  set(proj_DEPENDENCIES Boost gflags glog ProtoBuf-CMake ProtoBuf HDF5)
  if(NOT APPLE)
    list(APPEND proj_DEPENDENCIES OpenBLAS)
  endif()
  if(MITK_USE_OpenCV)
    list(APPEND proj_DEPENDENCIES OpenCV)
  endif()

  if(${NIFTK_USE_CUDA})
    set(CPU_ONLY OFF)
  else()
    set(CPU_ONLY ON)
  endif()

  set(_protobuf_args
    -DPROTOBUF_DIR:PATH=${ProtoBuf_DIR}
    -DPROTOBUF_INCLUDE_DIR:PATH=${ProtoBuf_INCLUDE_DIR}
    -DPROTOBUF_LIBRARY_DIR:PATH=${ProtoBuf_LIBRARY_DIR}
  )
  if (WIN32)
    list(APPEND _protobuf_args -DPROTOBUF_PROTOC_EXECUTABLE:FILEPATH=${ProtoBuf_DIR}/bin/protoc.exe)
  else()
    list(APPEND _protobuf_args -DPROTOBUF_PROTOC_EXECUTABLE:FILEPATH=${ProtoBuf_BUILD_DIR}/protoc/protoc)
  endif()

  set(_openblas_args)
  if(NOT APPLE)
    set(_openblas_args
      -DOpenBLAS_DIR:PATH=${OpenBLAS_DIR}
      -DOpenBLAS_INCLUDE_DIR:PATH=${OpenBLAS_INCLUDE_DIR}
      -DOpenBLAS_LIBRARY_DIR:PATH=${OpenBLAS_LIBRARY_DIR}
    )
  endif()
  if(WIN32)
    list(APPEND _openblas_args -DOpenBLAS_LIB:FILEPATH=${OpenBLAS_LIBRARY_DIR}/libopenblas.lib)
  else()
    list(APPEND _openblas_args -DOpenBLAS_LIB:FILEPATH=${OpenBLAS_LIBRARY_DIR}/libopenblas.so)
  endif()

  if (WIN32)
    set(CAFFE_CXX_FLAGS "-DNOMINMAX")
  endif()

  if(NOT DEFINED Caffe_DIR)

    ExternalProject_Add(${proj}
      LIST_SEPARATOR ^^
      PREFIX ${proj_CONFIG}
      SOURCE_DIR ${proj_SOURCE}
      BINARY_DIR ${proj_BUILD}
      INSTALL_DIR ${proj_INSTALL}
      URL ${proj_LOCATION}
      URL_MD5 ${proj_CHECKSUM}
      CMAKE_GENERATOR ${gen}
      CMAKE_ARGS
        ${EP_COMMON_ARGS}
        "-DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS} ${CAFFE_CXX_FLAGS}"
        -DCMAKE_PREFIX_PATH:PATH=${NifTK_PREFIX_PATH}
        -DNIFTK_BINARY_DIR:PATH=${CMAKE_BINARY_DIR}
        -DGFLAGS_DIR:PATH=${gflags_DIR}
        -DGFLAGS_INCLUDE_DIRS:PATH=${gflags_INCLUDE_DIR}
        -DGFLAGS_LIBRARY_DIR:PATH=${gflags_LIBRARY_DIR}
        -DGLOG_DIR:PATH=${glog_DIR}
        -DGLOG_INCLUDE_DIRS:PATH=${glog_INCLUDE_DIR}
        -DGLOG_LIBRARY_DIR:PATH=${glog_LIBRARY_DIR}
        ${_protobuf_args}
        ${_openblas_args}
        -DBoost_NO_SYSTEM_PATHS:BOOL=ON
        -DBoost_ADDITIONAL_VERSIONS:STRING=1.56
        -DHDF5_PREFIX:String=niftk
        -DHDF5_DIR:PATH=${HDF5_DIR}
        -DHDF5_INCLUDE_DIRS:STRING=${HDF5_INCLUDE_DIR}
        -DHDF5_HL_INCLUDE_DIR:PATH=${HDF5_INCLUDE_DIR}
        -DHDF5_LIBRARY_DIR:PATH=${HDF5_LIBRARY_DIR}
        -DUSE_OPENCV:BOOL=${MITK_USE_OpenCV}
        -DOpenCV_DIR:PATH=${OpenCV_DIR}
        -DUSE_LEVELDB:BOOL=OFF
        -DUSE_LMDB:BOOL=OFF
        -DBUILD_python:BOOL=OFF
        -DCPU_ONLY:BOOL=${CPU_ONLY}
      CMAKE_CACHE_ARGS
        ${EP_COMMON_CACHE_ARGS}
        -DBLAS:STRING=Open
      CMAKE_CACHE_DEFAULT_ARGS
        ${EP_COMMON_CACHE_DEFAULT_ARGS}
      DEPENDS ${proj_DEPENDENCIES}
    )

    set(Caffe_DIR ${proj_INSTALL})
    set(Caffe_SOURCE_DIR ${proj_SOURCE})
    set(NifTK_PREFIX_PATH ${proj_INSTALL}^^${NifTK_PREFIX_PATH})
    mitkFunctionInstallExternalCMakeProject(${proj})

    message("SuperBuild loading Caffe from ${Caffe_DIR}.")

  else(NOT DEFINED Caffe_DIR)

    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

  endif(NOT DEFINED Caffe_DIR)

endif()
