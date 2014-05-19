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

find_package(ITK)
if(ITK_FOUND)
  include("${MITK_MODULES_PACKAGE_DEPENDS_DIR}/MITK_ITK_Config.cmake")
  include(${ITK_USE_FILE})
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
    niftkCGALMesherBackEnd.cxx
    niftkMeditMeshParser.cxx
    niftkMeshGenerator.cxx
    niftkMeshMerger.cxx
    )

  target_link_libraries(niftkMeshingLib
    niftkITK
    niftkVTK
    niftkcommon
    ${CGAL_LIBRARIES}
    ${CGAL_3RD_PARTY_LIBRARIES}
    ${ITK_LIBRARIES}
    ${VTK_LIBRARIES}
    )

else (WIN32)
  add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/niftkCGALMesherBackEnd.o
    COMMAND ${CMAKE_CXX_COMPILER}
    ARGS -c -o ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/niftkCGALMesherBackEnd.o ${CMAKE_CURRENT_SOURCE_DIR}/niftkCGALMesherBackEnd.cxx
    ARGS -I${CMAKE_CURRENT_SOURCE_DIR}/../Exceptions -I${CGAL_INCLUDE_DIRS}
    ARGS `echo ${NIFTK_CGAL_CXX_FLAGS}`
    SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/niftkCGALMesherBackEnd.cxx
    )

  add_library(niftkMeshingLib
    ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/niftkCGALMesherBackEnd.o
    niftkMeditMeshParser.cxx
    niftkMeshGenerator.cxx
    niftkMeshMerger.cxx
    )
  target_link_libraries(niftkMeshingLib
    ${CGAL_LIBRARIES}
    ${CGAL_3RD_PARTY_LIBRARIES}
    niftkITK
    niftkVTK
    niftkcommon
    ${ITK_LIBRARIES}
    ${VTK_LIBRARIES}
    )
endif (WIN32)

install(TARGETS niftkMeshingLib
  LIBRARY DESTINATION ${NIFTK_INSTALL_LIB_DIR}
  ARCHIVE DESTINATION ${NIFTK_INSTALL_LIB_DIR}
  COMPONENT libraries
)
