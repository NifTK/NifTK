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

PROJECT(uk_ac_ucl_cmic_gui_qt_niftymidas)

MACRO_CREATE_MITK_CTK_PLUGIN(
  EXPORT_DIRECTIVE CMIC_QT_NIFTYMIDASAPP
  EXPORTED_INCLUDE_SUFFIXES src
  MODULE_DEPENDENCIES CTK niftkCoreGui
)
