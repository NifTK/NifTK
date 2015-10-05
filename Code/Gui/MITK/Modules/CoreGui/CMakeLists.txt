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
  INCLUDE_DIRS Events LookupTables IO Internal Rendering
  DEPENDS MitkCore MitkQtWidgets MitkQtWidgetsExt niftkCore
  PACKAGE_DEPENDS Qt4|QtGui Qt4|QtOpenGL CTK|CTKWidgets
)

target_link_libraries(niftkCoreGui PRIVATE niftkITK niftkVTK)

if (NIFTK_USE_COTIRE AND COMMAND cotire)
  cotire(niftkCoreGui)
endif()

if(BUILD_TESTING)
  #add_subdirectory(Testing)
endif(BUILD_TESTING)
