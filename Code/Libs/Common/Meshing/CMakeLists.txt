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

cmake_minimum_required(VERSION 2.6)

set(MEDIT_LIBS  
	vtkGraphics
  	vtkIO
 	vtkCommon
)

if(VTK_FOUND)
  include(${VTK_USE_FILE})
else(VTK_FOUND)  
  message(FATAL_ERROR
    "This program requires the VTK library, and will not be compiled.")  
endif(VTK_FOUND)	

if(ITK_FOUND)
  include(${ITK_USE_FILE})
else(ITK_FOUND)
  message(FATAL_ERROR
    "ITK not found. Please set ITK_DIR.")
endif(ITK_FOUND)

if(CGAL_FOUND)
 include(${CGAL_USE_FILE})
else(CGAL_FOUND)  
 message(FATAL_ERROR
   "This program requires the CGAL library, and will not be compiled.")  
endif(CGAL_FOUND)	

#### Build CGAL meshing module
if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  set(NIFTK_CGAL_CXX_FLAGS "${CMAKE_CXX_FLAGS_DEBUG};-frounding-math;-fPIC")
else ()  
  if (${CMAKE_BUILD_TYPE} STREQUAL "Release")
    set(NIFTK_CGAL_CXX_FLAGS "${CMAKE_CXX_FLAGS_RELEASE};-frounding-math;-fPIC")
  else ()
    set(NIFTK_CGAL_CXX_FLAGS "${CMAKE_CXX_FLAGS};-frounding-math;-fPIC")
  endif ()
endif ()

if (WIN32)  
  add_library(niftkMeshingLib
    CGALMesherBackEnd.cxx
    MeditMeshParser.cxx
    MeshGenerator.cxx
    MeshMerger.cxx
    )

  target_link_libraries(niftkMeshingLib
    CGAL-vc90-mt-gd
    CGAL_ImageIO-vc90-mt-gd
    niftkITK
    niftkVTK
    niftkcommon
    )

else (WIN32)  
  add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CGALMesherBackEnd.o
    COMMAND ${CMAKE_CXX_COMPILER} 
    ARGS -c -o ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CGALMesherBackEnd.o ${CMAKE_CURRENT_SOURCE_DIR}/CGALMesherBackEnd.cxx 
    ARGS -I${CMAKE_CURRENT_SOURCE_DIR}/../Exceptions -I${CGAL_INCLUDE_DIRS}
    ARGS `echo ${NIFTK_CGAL_CXX_FLAGS}`
    SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/CGALMesherBackEnd.cxx
    )

  add_library(niftkMeshingLib
    ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CGALMesherBackEnd.o
    MeditMeshParser.cxx
    MeshGenerator.cxx
    MeshMerger.cxx
    )
  target_link_libraries(niftkMeshingLib
    CGAL
    CGAL_ImageIO
    vtkHybrid
    niftkITK
    niftkVTK
    niftkcommon    
    )
endif (WIN32)

install(TARGETS niftkMeshingLib
  LIBRARY DESTINATION ${NIFTK_INSTALL_LIB_DIR} 
  ARCHIVE DESTINATION ${NIFTK_INSTALL_LIB_DIR} 
  COMPONENT libraries 
)
