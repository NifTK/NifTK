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

project(uk_ac_ucl_cmic_igipointreg)

mitk_create_plugin(
  EXPORT_DIRECTIVE POINTREG_EXPORT
  EXPORTED_INCLUDE_SUFFIXES src
  MODULE_DEPENDS niftkIGIGui niftkPointReg
)

if (NIFTK_USE_COTIRE AND COMMAND cotire)
  cotire(uk_ac_ucl_cmic_igipointreg)
endif()
