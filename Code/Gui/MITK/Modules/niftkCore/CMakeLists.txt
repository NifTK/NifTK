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

# Note:
# The PACKAGE_DEPENDS clause should contain ITK|ITKCommon+ITKIOImageBase+ITKIOTransformBase,
# but not ITK, i.e. the whole ITK package. The new way of specifying certain
# libraries of a package is supported by the new 'mitk_use_modules' function.
# With the old MITK_USE_MODULE macro you could specify only full packages.
# The MITK_CREATE_MODULE macro (below) is now implemented using this new function,
# but our NIFTK_CREATE_GUI_APPLICATION macro (what we use to create BlueBerry applications)
# still uses the MITK_USE_MODULE macro because the function did not collect the
# module library directories correctly, what resulted linker errors.
# Until this problem is resolved, we have to keep the full 'ITK' on the
# PACKAGE_DEPENDS list.

MITK_CREATE_MODULE(niftkCore
  EXPORT_DEFINE NIFTKCORE_EXPORT
  INCLUDE_DIRS Algorithms Common Controllers DataManagement DataNodeProperties Interactions IO Rendering
  DEPENDS MitkCore
  PACKAGE_DEPENDS ITK ITK|ITKIOTransformBase VTK|vtkRenderingVolumeOpenGL
)

target_link_libraries(niftkCore
  niftkcommon
  niftkITK
  niftkVTK
)

if (NIFTK_USE_COTIRE AND COMMAND cotire)
  cotire(niftkCore)
endif()

if(BUILD_TESTING)
  add_subdirectory(Testing)
endif()
