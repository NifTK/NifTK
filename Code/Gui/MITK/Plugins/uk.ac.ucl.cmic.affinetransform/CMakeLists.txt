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

project(uk_ac_ucl_cmic_affinetransform)

# Temporary workaround until the qRestAPI lib directory is not exposed properly to
# projects depending on CTK. Should be removed as soon as this is fixed in CTK.
link_directories(${CTK_DIR}/qRestAPI-build)

MACRO_CREATE_MITK_CTK_PLUGIN(
  EXPORT_DIRECTIVE AFFINETRANSFORM_EXPORT
  EXPORTED_INCLUDE_SUFFIXES src
  MODULE_DEPENDENCIES QmitkExt niftkCore
)
