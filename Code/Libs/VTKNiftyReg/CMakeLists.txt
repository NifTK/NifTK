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

if(NOT ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
    # OpenMP is disabled when the compiler is based on Clang
    find_package(OpenMP)
    if(OPENMP_FOUND)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif(OPENMP_FOUND)
endif(NOT ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")

add_definitions(-DUSE_NIFTYREG)
link_directories(${NIFTYREG_LIBRARY_DIR})
include_directories(${NIFTYREG_INCLUDE_DIR})

set(niftkVTKNiftyReg_SRCS
  niftkF3DControlGridToVTKPolyData.cxx
  )

add_library(niftkVTKNiftyReg ${niftkVTKNiftyReg_SRCS})

target_link_libraries(niftkVTKNiftyReg 
  niftkcommon
  ${NIFTK_VTK_LIBS_BUT_WITHOUT_QT}
  ${Boost_LIBRARIES}
  ITKniftiio
  ${NIFTYREG_LIBRARIES}
  )

if(BUILD_SHARED_LIBS)
  if(WIN32)
    add_definitions(-DNIFTKVTKNIFTYREG_WINDOWS_EXPORT)
  endif(WIN32)
endif(BUILD_SHARED_LIBS)

MITK_INSTALL(TARGETS niftkVTKNiftyReg)
