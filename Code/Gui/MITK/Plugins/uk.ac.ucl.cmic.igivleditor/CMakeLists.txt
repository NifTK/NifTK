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

project(uk_ac_ucl_cmic_igivleditor)

# FIXME: this should go away: vlwidget should hide all the vl stuff behind pimpl
find_package(VL COMPONENTS VLCore VLGraphics VLVolume VLQt4 REQUIRED)
include_directories(${VL_INCLUDE_DIRS})

MACRO_CREATE_MITK_CTK_PLUGIN(
  EXPORT_DIRECTIVE UK_AC_UCL_CMIC_IGIVLEDITOR
  EXPORTED_INCLUDE_SUFFIXES src
  MODULE_DEPENDS MitkQtWidgets niftkVL
  )

# Cotire must be ON on Windows and Mac and must be OFF on Linux.
if (COMMAND cotire AND NOT ${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  cotire(uk_ac_ucl_cmic_igivleditor)
endif()
