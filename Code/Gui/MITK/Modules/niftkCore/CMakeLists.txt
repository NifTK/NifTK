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

MITK_CREATE_MODULE(niftkCore
  EXPORT_DEFINE NIFTKCORE_EXPORT
  INCLUDE_DIRS Algorithms Common Controllers DataManagement DataNodeProperties Interactions IO Rendering
  DEPENDS MitkExt 
  PACKAGE_DEPENDS ITK VTK
)

target_link_libraries(niftkCore
  niftkcommon
  niftkITK
  niftkVTK
  )
  
if(BUILD_TESTING)
  add_subdirectory(Testing)
endif()
