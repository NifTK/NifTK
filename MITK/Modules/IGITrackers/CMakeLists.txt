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
  DEPENDS MitkIGT niftkIGIDataSources niftkcommon
  PACKAGE_DEPENDS Qt4|QtGui Qt5|Widgets
)

target_link_libraries(niftkIGITrackers)

include_directories()

if (NIFTK_USE_COTIRE AND COMMAND cotire)
  cotire(niftkIGITrackers)
endif()

if(BUILD_TESTING)
  add_subdirectory(Testing)
endif()
