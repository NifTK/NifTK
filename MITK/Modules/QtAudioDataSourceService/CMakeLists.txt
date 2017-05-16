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

set(_got_multimedia OFF)
if (MITK_USE_Qt5 AND Qt5Multimedia_DIR)
  set(_got_multimedia ON)
elseif(MITK_USE_Qt4 AND QT_QTMULTIMEDIA_INCLUDE_DIR)
  set(_got_multimedia ON)
endif()

if (${_got_multimedia})

  MITK_CREATE_MODULE(
    DEPENDS niftkIGIDataSources
    PACKAGE_DEPENDS Qt4|QtCore+QtMultimedia Qt5|Widgets+Multimedia  NiftyLink|OpenIGTLink
    AUTOLOAD_WITH niftkIGIDataSources
  )

  if (NIFTK_USE_COTIRE AND COMMAND cotire)
    cotire(niftkQtAudioDataSourceService)
  endif()

endif()
