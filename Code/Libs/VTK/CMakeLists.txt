#/*================================================================================
#
#  NifTK: An image processing toolkit jointly developed by the
#              Dementia Research Centre, and the Centre For Medical Image Computing
#              at University College London.
#  
#  See:        http://dementia.ion.ucl.ac.uk/
#              http://cmic.cs.ucl.ac.uk/
#              http://www.ucl.ac.uk/
#
#  Copyright (c) UCL : See LICENSE.txt in the top level directory for details. 
#
#  Last Changed      : $LastChangedDate: 2010-07-28 12:37:56 +0100 (Wed, 28 Jul 2010) $ 
#  Revision          : $Revision: 8038 $
#  Last modified by  : $Author: mjc $
#
#  Original author   : m.clarkson@ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/

SET(niftkVTK_SRCS
  Common/vtkFunctions.cpp
)

if(NOT ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
    # OpenMP is disabled when the compiler is based on Clang
    find_package(OpenMP)
    if(OPENMP_FOUND)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif(OPENMP_FOUND)
endif(NOT ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")

IF(FALSE AND NIFTYREG_FOUND)
  ADD_DEFINITIONS(-DUSE_NIFTYREG)
  LINK_DIRECTORIES(${NIFTYREG_LIBRARY_DIR})
  INCLUDE_DIRECTORIES(${NIFTYREG_INCLUDE_DIR})
      
  SET(niftkVTK_SRCS
    ${niftkVTK_SRCS}
    niftkF3DControlGridToVTKPolyData.cxx
    )
ENDIF(FALSE AND NIFTYREG_FOUND)

ADD_LIBRARY(niftkVTK ${niftkVTK_SRCS})

IF(FALSE AND NIFTYREG_FOUND)
  TARGET_LINK_LIBRARIES(niftkVTK
    niftkcommon
    ${NIFTK_VTK_LIBS_BUT_WITHOUT_QT}
    ${Boost_LIBRARIES}  
    ${NIFTYREG_LIBRARIES}
    ) 

ELSE(FALSE AND NIFTYREG_FOUND)

  TARGET_LINK_LIBRARIES(niftkVTK
    niftkcommon
    ${NIFTK_VTK_LIBS_BUT_WITHOUT_QT}
    ${Boost_LIBRARIES}  
    ) 
ENDIF(FALSE AND NIFTYREG_FOUND)

IF(BUILD_SHARED_LIBS)
  IF(WIN32)
    ADD_DEFINITIONS(-DNIFTKVTK_WINDOWS_EXPORT)
  ENDIF(WIN32)
ENDIF(BUILD_SHARED_LIBS)

MITK_INSTALL(TARGETS niftkVTK)
