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

set(ITK_NO_IO_FACTORY_REGISTER_MANAGER 0)
include(${MITK_MODULES_PACKAGE_DEPENDS_DIR}/MITK_ITK_Config.cmake)

add_subdirectory(Common)
add_subdirectory(ITK)

if(VTK_FOUND)
  add_subdirectory(VTK)
endif(VTK_FOUND)

