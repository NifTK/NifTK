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
if(NOT WIN32 AND NOT APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frounding-math -fPIC")
endif(NOT WIN32 AND NOT APPLE)

add_library(niftkMeshing
  niftkCGALMesherBackEnd.cxx
  niftkMeditMeshParser.cxx
  niftkMeshGenerator.cxx
  niftkMeshMerger.cxx
  )

target_link_libraries(niftkMeshing
  niftkITK
  niftkVTK
  niftkcommon
  ${CGAL_LIBRARIES}
  ${CGAL_3RD_PARTY_LIBRARIES}
  ${ITK_LIBRARIES}
  ${VTK_LIBRARIES}
  )

install(TARGETS niftkMeshing
  LIBRARY DESTINATION ${NIFTK_INSTALL_LIB_DIR}
  ARCHIVE DESTINATION ${NIFTK_INSTALL_LIB_DIR}
  COMPONENT libraries
)
