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

MITK_CREATE_MODULE(
  INCLUDE_DIRS Common OverlayEditor
  DEPENDS MitkQtWidgetsExt MitkIGTUI niftkCoreGui niftkIGI
  PACKAGE_DEPENDS Qt4|QtGui QVTK NiftyLink
)

if (NIFTK_USE_COTIRE AND COMMAND cotire)
  cotire(niftkIGIGui)
endif()

if(BUILD_TESTING)
  add_subdirectory(Testing)
endif()
