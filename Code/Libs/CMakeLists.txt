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

add_subdirectory(Common)

find_package(ITK)
if(ITK_FOUND)
  include("${MITK_MODULES_PACKAGE_DEPENDS_DIR}/MITK_ITK_Config.cmake")
  add_subdirectory(ITK)
endif(ITK_FOUND)

if(VTK_FOUND)
  add_subdirectory(VTK)

  if(NIFTYREG_FOUND)
    add_subdirectory(VTKNiftyReg)
  endif(NIFTYREG_FOUND)

endif(VTK_FOUND)

if(NVAPI_FOUND)
  add_subdirectory(NVAPI)
endif(NVAPI_FOUND)
