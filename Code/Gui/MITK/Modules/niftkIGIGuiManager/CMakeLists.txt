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

set(niftkIGIGuiManager_DEPENDS
  niftkIGIGui
)

if(NVAPI_FOUND)
  set(niftkIGIGuiManager_DEPENDS
    ${niftkIGIGuiManager_DEPENDS}
    niftkNVidia
    niftkNVidiaGui
  )
endif()

MITK_CREATE_MODULE(niftkIGIGuiManager
  EXPORT_DEFINE NIFTKIGIGUIMANAGER_EXPORT
  INCLUDE_DIRS DataSources
  DEPENDS ${niftkIGIGuiManager_DEPENDS}
  PACKAGE_DEPENDS QT
  QT_MODULE 
)

if(BUILD_TESTING)  
  #add_subdirectory(Testing)
endif()
