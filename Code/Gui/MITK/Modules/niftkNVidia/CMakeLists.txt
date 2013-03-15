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

MITK_CREATE_MODULE(niftkNVidia
  EXPORT_DEFINE NIFTKNVIDIA_EXPORT
  INCLUDE_DIRS DataManagement
  DEPENDS niftkIGI
)

TARGET_LINK_LIBRARIES(niftkNVidia
  niftknvapi
  )

IF(BUILD_TESTING)  
  ADD_SUBDIRECTORY(Testing)
ENDIF()