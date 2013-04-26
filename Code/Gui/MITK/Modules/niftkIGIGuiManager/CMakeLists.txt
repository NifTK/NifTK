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

SET(niftkIGIGuiManager_DEPENDS
  niftkIGIGui
)

IF(NVAPI_FOUND)
  SET(niftkIGIGuiManager_DEPENDS
    ${niftkIGIGuiManager_DEPENDS}
    niftkNVidia
    niftkNVidiaGui
  )
ENDIF()

MITK_CREATE_MODULE(niftkIGIGuiManager
  EXPORT_DEFINE NIFTKIGIGUIMANAGER_EXPORT
  INCLUDE_DIRS DataSources
  DEPENDS ${niftkIGIGuiManager_DEPENDS}
  PACKAGE_DEPENDS QT
  QT_MODULE 
)

IF(BUILD_TESTING)  
  #ADD_SUBDIRECTORY(Testing)
ENDIF()
