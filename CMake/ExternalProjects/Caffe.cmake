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

set(version "rc3")
set(location "${NIFTK_EP_TARBALL_LOCATION}/caffe-${version}.tar.gz")

niftkMacroDefineExternalProjectVariables(Caffe ${version} ${location})
set(proj_DEPENDENCIES OpenBLAS Boost ProtoBuf GLog GFlags HDF5)

if(${NIFTK_USE_CUDA})
  set(CPU-ONLY OFF)
else(${NIFTK_USE_CUDA})
  set(CPU-ONLY ON)
endif(${NIFTK_USE_CUDA})

if(NOT DEFINED Caffe_DIR)

  ExternalProject_Add(${proj}
    LIST_SEPARATOR ^^
    PREFIX ${proj_CONFIG}
    SOURCE_DIR ${proj_SOURCE}
    BINARY_DIR ${proj_BUILD}
    INSTALL_DIR ${proj_INSTALL}
    URL ${proj_LOCATION}
    URL_MD5 ${proj_CHECKSUM}
    #CONFIGURE_COMMAND ""
    #UPDATE_COMMAND ""
    #BUILD_COMMAND ""
    #INSTALL_COMMAND ""
    CMAKE_GENERATOR ${gen}
    CMAKE_ARGS
      ${EP_COMMON_ARGS}
      -DCMAKE_PREFIX_PATH:PATH=${NifTK_PREFIX_PATH}
      -DZLIB_INCLUDE_DIR:PATH=${ZLIB_INCLUDE_DIR}
      -DZLIB_LIBRARY:FILEPATH=${ZLIB_LIBRARY}
      #-DBOOST_ROOT:PATH=${BOOST_ROOT}
      #-DBOOST_INCLUDEDIR:PATH=${BOOST_ROOT}/include
      #-DBOOST_LIBRARYDIR:PATH=${BOOST_ROOT}/lib
      #-DBoost_DIR:PATH=${BOOST_ROOT}
      #-DBoost_INCLUDE_DIRS:PATH=${BOOST_INCLUDEDIR}
      #-DBoost_LIBRARY_DIRS:PATH=${BOOST_LIBRARYDIR}
      -DGFLAGS_DIR:PATH=${GFlags_DIR}
      -DGFLAGS_INCLUDE_DIR:PATH=${GFlags_INCLUDE_DIR}
      -DGFLAGS_LIBRARY_DIR:PATH=${GFlags_LIBRARY_DIR}
      -DGFLAGS_LIBRARY:PATH=${GFlags_LIBRARY}
      -DGLOG_DIR:PATH=${GLog_DIR}
      -DGLOG_INCLUDE_DIR:PATH=${GLog_INCLUDE_DIR}
      -DGLOG_LIBRARY_DIR:PATH=${GLog_LIBRARY_DIR}
      -DGLOG_LIBRARY:PATH=${GLog_LIBRARY}
      -DOpenBLAS_DIR:PATH=${OpenBLAS_DIR}
      -DOpenBLAS_INCLUDE_DIR:PATH=${OpenBLAS_INCLUDE_DIR}
      -DOpenBLAS_LIBRARY_DIR:PATH=${OpenBLAS_LIBRARY_DIR}
      -DOpenBLAS_LIB:PATH=${OpenBLAS_LIBRARY}
      -DPROTOBUF_DIR:PATH=${ProtoBuf_DIR}
      -DPROTOBUF_INCLUDE_DIR:PATH=${ProtoBuf_INCLUDE_DIR}
      -DPROTOBUF_LIBRARY_DIR:PATH=${ProtoBuf_LIBRARY_DIR}
      -DPROTOBUF_LIBRARY:PATH=${ProtoBuf_LIBRARY}
      -DPROTOBUF_PROTOC_EXECUTABLE=${ProtoBuf_PROTOC_EXECUTABLE}
      -DHDF5_DIR:PATH=${HDF5_DIR}
      #-DHDF5_ROOT:PATH=${HDF5_DIR}
      #-DHDF5_ROOT_DIR:PATH=${HDF5_DIR}
      #-DHDF5_INCLUDE_DIRS:STRING=${HDF5_INCLUDE_DIR}
      #-DHDF5_LIBRARY_DIR:PATH=${HDF5_LIBRARY_DIR}
      #-DHDF5_LIBRARY_DIRS:PATH=${HDF5_LIBRARY_DIR}
      #-DHDF5_CONFIG_DIR:PATH=${HDF5_DIR}/share/cmake
      #-DHDF5_CONFIG_DIR_HINT:PATH=${HDF5_DIR}/share/cmake
      #-DHDF5_ROOT_DIR_HINT:PATH=${HDF5_DIR}/share/cmake
      -DUSE_OPENCV:BOOL=${MITK_USE_OpenCV}
      -DOpenCV_DIR:PATH=${OpenCV_DIR}
      -DUSE_LEVELDB:BOOL=OFF
      -DUSE_LMDB:BOOL=OFF
      -DBUILD_python:BOOL=OFF
      -DCPU-ONLY:BOOL=${CPU-ONLY}
    CMAKE_CACHE_ARGS
      ${EP_COMMON_CACHE_ARGS}
      -DBLAS:STRING=Open
      -DBoost_DIR:PATH=${BOOST_ROOT}
      -DBoost_INCLUDE_DIR:PATH=${BOOST_ROOT}/include
      -DBoost_LIBRARY_DIR:PATH=${BOOST_ROOT}/lib
    CMAKE_CACHE_DEFAULT_ARGS
      ${EP_COMMON_CACHE_DEFAULT_ARGS}
    DEPENDS ${proj_DEPENDENCIES}
  )

  set(Caffe_SOURCE_DIR ${proj_SOURCE})
  set(Caffe_DIR ${proj_INSTALL})
  set(Caffe_INCLUDE_DIR ${Caffe_DIR}/include)
  set(Caffe_LIBRARY_DIR ${Caffe_DIR}/lib)

  mitkFunctionInstallExternalCMakeProject(${proj})

  message("SuperBuild loading Caffe from ${Caffe_DIR}.")

else(NOT DEFINED Caffe_DIR)

  mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

endif(NOT DEFINED Caffe_DIR)
