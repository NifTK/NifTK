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

SET(niftkVTK_SRCS
  Common/vtkFunctions.cpp
  niftkVTKIterativeClosestPoint.cxx
  niftkvtk4PointsReader.cxx
)

if(NOT ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
    # OpenMP is disabled when the compiler is based on Clang
    find_package(OpenMP)
    if(OPENMP_FOUND)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif(OPENMP_FOUND)
endif(NOT ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")

IF(NIFTYREG_FOUND)
  ADD_DEFINITIONS(-DUSE_NIFTYREG)
  LINK_DIRECTORIES(${NIFTYREG_LIBRARY_DIR})
  INCLUDE_DIRECTORIES(${NIFTYREG_INCLUDE_DIR})
      
  SET(niftkVTK_SRCS
    ${niftkVTK_SRCS}
    niftkF3DControlGridToVTKPolyData.cxx
    )
ENDIF(NIFTYREG_FOUND)

ADD_LIBRARY(niftkVTK ${niftkVTK_SRCS})

IF(NIFTYREG_FOUND)
  TARGET_LINK_LIBRARIES(niftkVTK
    niftkcommon
    ${NIFTK_VTK_LIBS_BUT_WITHOUT_QT}
    ${Boost_LIBRARIES}  
    ${NIFTYREG_LIBRARIES}
    ) 

ELSE(NIFTYREG_FOUND)

  TARGET_LINK_LIBRARIES(niftkVTK
    niftkcommon
    ${NIFTK_VTK_LIBS_BUT_WITHOUT_QT}
    ${Boost_LIBRARIES}  
    ) 
ENDIF(NIFTYREG_FOUND)

IF(BUILD_SHARED_LIBS)
  IF(WIN32)
    ADD_DEFINITIONS(-DNIFTKVTK_WINDOWS_EXPORT)
  ENDIF(WIN32)
ENDIF(BUILD_SHARED_LIBS)

MITK_INSTALL(TARGETS niftkVTK)
