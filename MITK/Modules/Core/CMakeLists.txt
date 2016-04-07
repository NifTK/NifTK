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
  INCLUDE_DIRS Algorithms Common DataManagement DataNodeProperties Interactions IO LookupTables Rendering
  DEPENDS MitkCore MitkDataTypesExt
  PACKAGE_DEPENDS  ITK|ITKIOTransformBase VTK|vtkRenderingVolumeOpenGL Boost|boost_system Qt4|QtCore Qt5|Core
)

target_link_libraries(niftkCore PRIVATE niftkcommon niftkITK niftkVTK)

if (NIFTK_USE_COTIRE AND COMMAND cotire)
  cotire(niftkCore)
endif()

if(BUILD_TESTING)
  add_subdirectory(Testing)
endif()
