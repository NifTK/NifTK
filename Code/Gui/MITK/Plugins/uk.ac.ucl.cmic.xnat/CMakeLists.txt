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

#find_package(qRestAPI REQUIRED)
#include(${qRestAPI_USE_FILE})                   # Call include_directories

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/internal)

#link_directories(${CMAKE_PROJECT_BINARY_DIR}/qRestAPI-build)
#message(link_directories(${CMAKE_PROJECT_BINARY_DIR}/qRestAPI-build))

MACRO_CREATE_MITK_CTK_PLUGIN(
  EXPORT_DIRECTIVE XNAT_EXPORTS
  EXPORTED_INCLUDE_SUFFIXES src
  MODULE_DEPENDENCIES Mitk MitkExt Qmitk QmitkExt XnatRest CTK
)
