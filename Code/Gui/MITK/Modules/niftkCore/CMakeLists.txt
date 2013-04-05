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
  PACKAGE_DEPENDS ITK
)

TARGET_LINK_LIBRARIES(niftkCore
  niftkcommon
  niftkITK
  )
  
IF(BUILD_TESTING)
  ADD_SUBDIRECTORY(Testing)
ENDIF()
