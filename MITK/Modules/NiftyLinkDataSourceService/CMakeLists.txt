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
    DEPENDS niftkCore niftkIGIDataSources niftkQImageConversion
    PACKAGE_DEPENDS Qt4|QtCore Qt5|Widgets+Network NiftyLink
    AUTOLOAD_WITH niftkIGIServices
  )

  if (NIFTK_USE_COTIRE AND COMMAND cotire)
    cotire(niftkNiftyLinkDataSourceService)
  endif()

endif()
