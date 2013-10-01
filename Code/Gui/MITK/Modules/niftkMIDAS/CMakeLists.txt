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

MITK_CREATE_MODULE(niftkMIDAS
  EXPORT_DEFINE NIFTKMIDAS_EXPORT
  INCLUDE_DIRS Algorithms Common Controllers DataManagement DataNodeProperties Interactions IO Rendering
  DEPENDS niftkCore Segmentation niftkDnDDisplay
  PACKAGE_DEPENDS ITK
)

target_link_libraries(niftkMIDAS
  niftkcommon
  niftkITK
  )
  
if(BUILD_TESTING)
  add_subdirectory(Testing)
endif()
