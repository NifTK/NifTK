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

if(MITK_USE_OpenCV)

  MITK_CREATE_MODULE(
    DEPENDS niftkIGIDataSources niftkOpenCVImageConversion MitkOpenCVVideoSupport
    PACKAGE_DEPENDS Qt4|QtCore OpenCV
    AUTOLOAD_WITH niftkIGIServices
  )

  if (NIFTK_USE_COTIRE AND COMMAND cotire)
    cotire(niftkOpenCVVideoDataSourceService)
  endif()

endif()
