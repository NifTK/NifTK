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

ADD_DEFINITIONS(-DUSE_NIFTYREG)
LINK_DIRECTORIES(${NIFTYREG_LIBRARY_DIR})
INCLUDE_DIRECTORIES(${NIFTYREG_INCLUDE_DIR})

SET(niftkVTKNiftyReg_SRCS
  niftkF3DControlGridToVTKPolyData.cxx
  )

ADD_LIBRARY(niftkVTKNiftyReg ${niftkVTKNiftyReg_SRCS})

TARGET_LINK_LIBRARIES(niftkVTKNiftyReg 
  niftkcommon
  ${NIFTK_VTK_LIBS_BUT_WITHOUT_QT}
  ${Boost_LIBRARIES}
  ${NIFTYREG_LIBRARIES}
  )

IF(BUILD_SHARED_LIBS)
  IF(WIN32)
    ADD_DEFINITIONS(-DNIFTKVTKNIFTYREG_WINDOWS_EXPORT)
  ENDIF(WIN32)
ENDIF(BUILD_SHARED_LIBS)

MITK_INSTALL(TARGETS niftkVTKNiftyReg)
