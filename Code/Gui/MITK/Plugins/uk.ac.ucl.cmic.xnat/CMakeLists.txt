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

project(uk_ac_ucl_cmic_xnat)

find_package(Qt4 4.6.2 COMPONENTS QtScript REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/internal)

MACRO_CREATE_MITK_CTK_PLUGIN(
  EXPORT_DIRECTIVE XNAT_EXPORTS
  EXPORTED_INCLUDE_SUFFIXES src
  MODULE_DEPENDENCIES Mitk MitkExt Qmitk QmitkExt XnatRest CTK
)
