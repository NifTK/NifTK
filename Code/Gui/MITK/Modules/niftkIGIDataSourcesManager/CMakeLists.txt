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

set(niftkIGIDataSourcesManager_DEPENDS
  niftkIGIDataSources
)

if(NVAPI_FOUND AND CUDA_FOUND AND NIFTK_USE_CUDA)
  set(niftkIGIDataSourcesManager_DEPENDS
    ${niftkIGIDataSourcesManager_DEPENDS}
    niftkNVidia
    niftkNVidiaGui
  )
endif()

MITK_CREATE_MODULE(niftkIGIDataSourcesManager
  EXPORT_DEFINE NIFTKIGIGUIMANAGER_EXPORT
  INCLUDE_DIRS DataSources
  DEPENDS ${niftkIGIDataSourcesManager_DEPENDS}
  PACKAGE_DEPENDS Qt4|QtGui
)

if (NIFTK_USE_COTIRE AND COMMAND cotire)
  cotire(niftkIGIDataSourcesManager)
endif()

if(BUILD_TESTING)  
  #add_subdirectory(Testing)
endif()
