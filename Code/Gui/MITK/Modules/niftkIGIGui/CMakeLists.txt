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

set(_packages Qt4|QtGui QVTK NiftyLink)

# optional audio data source depends on qt-multimedia, which may not be available.
if(QT_QTMULTIMEDIA_INCLUDE_DIR)
  set(_packages
    ${_packages}
    Qt4|QtMultimedia
  )
endif()

MITK_CREATE_MODULE(niftkIGIGui
  EXPORT_DEFINE NIFTKIGIGUI_EXPORT
  INCLUDE_DIRS Common DataSources OverlayEditor
  DEPENDS MitkQtWidgetsExt MitkIGTUI niftkCoreGui niftkIGI niftkOpenCV
  PACKAGE_DEPENDS ${_packages}
)

if (NIFTK_USE_COTIRE AND COMMAND cotire)
  cotire(niftkIGIGui)
endif()

if(BUILD_TESTING)
  add_subdirectory(Testing)
endif()
