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

project(uk_ac_ucl_cmic_sideview)

MACRO_CREATE_MITK_CTK_PLUGIN(
  EXPORT_DIRECTIVE CMIC_QT_SIDEVIEW
  EXPORTED_INCLUDE_SUFFIXES src
  MODULE_DEPENDENCIES CTK niftkMIDASGui
)

if (NIFTK_USE_COTIRE AND COMMAND cotire)
  cotire(uk_ac_ucl_cmic_sideview)
endif()
