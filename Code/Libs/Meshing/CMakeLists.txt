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

set(ITK_NO_IO_FACTORY_REGISTER_MANAGER 0)
include(${MITK_MODULES_PACKAGE_DEPENDS_DIR}/MITK_ITK_Config.cmake)

if(CGAL_FOUND)
 include(${CGAL_USE_FILE})
else(CGAL_FOUND)
 message(FATAL_ERROR
   "This program requires the CGAL library, and will not be compiled.")
endif(CGAL_FOUND)


if(BUILD_SHARED_LIBS)
  if(WIN32)
    # remember https://cmiclab.cs.ucl.ac.uk/CMIC/NifTK/issues/3647
    add_definitions(-DNIFTKCOMMON_WINDOWS_EXPORT)
  endif(WIN32)
endif(BUILD_SHARED_LIBS)


#### Build CGAL meshing module
if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  # NOTE: NIFTK_CGAL_CXX_FLAGS only applies to non-windows
  set(NIFTK_CGAL_CXX_FLAGS "${CMAKE_CXX_FLAGS_DEBUG};-frounding-math;-fPIC")
else ()
  if (${CMAKE_BUILD_TYPE} STREQUAL "Release")
    set(NIFTK_CGAL_CXX_FLAGS "${CMAKE_CXX_FLAGS_RELEASE};-frounding-math;-fPIC")
  else ()
    set(NIFTK_CGAL_CXX_FLAGS "${CMAKE_CXX_FLAGS};-frounding-math;-fPIC")
  endif ()
endif ()

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

install(TARGETS niftkMeshingLib
  LIBRARY DESTINATION ${NIFTK_INSTALL_LIB_DIR}
  ARCHIVE DESTINATION ${NIFTK_INSTALL_LIB_DIR}
  COMPONENT libraries
)
