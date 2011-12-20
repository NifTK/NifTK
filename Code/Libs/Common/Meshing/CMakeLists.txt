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
#  Last Changed      : $LastChangedDate$ 
#  Revision          : $LastChangedRevision$
#  Last modified by  : $LastChangedBy$
#
#  Original author   : stian.johnsen.09@ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/

CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

SET(MEDIT_LIBS  
	vtkGraphics
  	vtkIO
 	vtkCommon
)

IF(VTK_FOUND)
  INCLUDE(${VTK_USE_FILE})
ELSE(VTK_FOUND)  
  MESSAGE(FATAL_ERROR
    "This program requires the VTK library, and will not be compiled.")  
ENDIF(VTK_FOUND)	

IF(ITK_FOUND)
  INCLUDE(${ITK_USE_FILE})
ELSE(ITK_FOUND)
  MESSAGE(FATAL_ERROR
    "ITK not found. Please set ITK_DIR.")
ENDIF(ITK_FOUND)

IF(CGAL_FOUND)
 INCLUDE(${CGAL_USE_FILE})
ELSE(CGAL_FOUND)  
 MESSAGE(FATAL_ERROR
   "This program requires the CGAL library, and will not be compiled.")  
ENDIF(CGAL_FOUND)	

#### Build CGAL meshing module
IF (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  SET(NIFTK_CGAL_CXX_FLAGS "${CMAKE_CXX_FLAGS_DEBUG};-frounding-math;-fPIC")
ELSE ()  
  IF (${CMAKE_BUILD_TYPE} STREQUAL "Release")
    SET(NIFTK_CGAL_CXX_FLAGS "${CMAKE_CXX_FLAGS_RELEASE};-frounding-math;-fPIC")
  ELSE ()
    SET(NIFTK_CGAL_CXX_FLAGS "${CMAKE_CXX_FLAGS};-frounding-math;-fPIC")
  ENDIF ()
ENDIF ()

IF (WIN32)  
  ADD_LIBRARY(niftkMeshingLib
    CGALMesherBackEnd.cxx
    MeditMeshParser.cxx
    MeshGenerator.cxx
    MeshMerger.cxx
    )

  TARGET_LINK_LIBRARIES(niftkMeshingLib
    CGAL-vc90-mt-gd
    CGAL_ImageIO-vc90-mt-gd
    niftkITK
    niftkVTK
    niftkcommon
    )

ELSE (WIN32)  
  ADD_CUSTOM_COMMAND(
    OUTPUT ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CGALMesherBackEnd.o
    COMMAND ${CMAKE_CXX_COMPILER} 
    ARGS -c -o ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CGALMesherBackEnd.o ${CMAKE_CURRENT_SOURCE_DIR}/CGALMesherBackEnd.cxx 
    ARGS -I${CMAKE_CURRENT_SOURCE_DIR}/../Exceptions -I${CGAL_INCLUDE_DIRS}
    ARGS `echo ${NIFTK_CGAL_CXX_FLAGS}`
    SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/CGALMesherBackEnd.cxx
    )

  ADD_LIBRARY(niftkMeshingLib
    ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CGALMesherBackEnd.o
    MeditMeshParser.cxx
    MeshGenerator.cxx
    MeshMerger.cxx
    )
  TARGET_LINK_LIBRARIES(niftkMeshingLib
    CGAL
    CGAL_ImageIO
    vtkHybrid
    niftkITK
    niftkVTK
    niftkcommon    
    )
ENDIF (WIN32)

INSTALL(TARGETS niftkMeshingLib
  LIBRARY DESTINATION ${NIFTK_INSTALL_LIB_DIR} 
  ARCHIVE DESTINATION ${NIFTK_INSTALL_LIB_DIR} 
  COMPONENT libraries 
)
