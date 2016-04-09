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

find_package(NiftyLink)
if (NiftyLink_FOUND)

  MITK_CREATE_MODULE(
    DEPENDS MitkCore niftkCore niftkIGIDataSources niftkOpenCVImageConversion
    PACKAGE_DEPENDS Qt4|QtCore Qt5|Widgets+Network NiftyLink OpenCV
    AUTOLOAD_WITH niftkIGI
  )

  if (NIFTK_USE_COTIRE AND COMMAND cotire)
    cotire(niftkMITKTrackerDataSourceService)
  endif()

endif()
