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

# Temporary workaround for CTK bug of not exposing external project library dirs.
# Should be removed as soon as this is fixed in CTK. (espakm)
include(${CTK_DIR}/qRestAPI-build/qRestAPIConfig.cmake)
link_directories(${qRestAPI_LIBRARY_DIRS})

MACRO_CREATE_MITK_CTK_PLUGIN(
  EXPORT_DIRECTIVE AFFINETRANSFORM_EXPORT
  EXPORTED_INCLUDE_SUFFIXES src
  MODULE_DEPENDENCIES QmitkExt niftkCore
)
