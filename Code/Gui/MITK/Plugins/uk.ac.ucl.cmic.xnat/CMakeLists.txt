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

mitk_create_plugin(
  EXPORT_DIRECTIVE XNAT_EXPORTS
  EXPORTED_INCLUDE_SUFFIXES src
  PACKAGE_DEPENDS CTK|CTKXNATCore+CTKXNATWidgets+CTKWidgets
  MODULE_DEPENDS MitkCore MitkQtWidgets MitkQtWidgetsExt MitkLegacyIO
)

if (NIFTK_USE_COTIRE AND COMMAND cotire)
  cotire(uk_ac_ucl_cmic_xnat)
endif()
