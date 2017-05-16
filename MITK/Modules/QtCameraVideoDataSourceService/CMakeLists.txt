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

if(MITK_USE_Qt5)

  MITK_CREATE_MODULE(
    DEPENDS niftkIGIDataSources niftkQImageConversion
    PACKAGE_DEPENDS Qt5|Widgets+Multimedia NiftyLink|OpenIGTLink
    AUTOLOAD_WITH niftkIGIDataSources
  )

  if (NIFTK_USE_COTIRE AND COMMAND cotire)
    cotire(niftkQtCameraVideoDataSourceService)
  endif()

endif()
