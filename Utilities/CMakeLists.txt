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

include(${MITK_MODULES_PACKAGE_DEPENDS_DIR}/MITK_ITK_Config.cmake)

add_executable(niftkFixIncludeGuards niftkFixIncludeGuards.cxx)
target_link_libraries(niftkFixIncludeGuards itksys)
