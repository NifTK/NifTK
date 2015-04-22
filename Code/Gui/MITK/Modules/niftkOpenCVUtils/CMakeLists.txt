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


MITK_CREATE_MODULE(niftkOpenCVUtils
  EXPORT_DEFINE NIFTKOPENCVUTILS_EXPORT
  DEPENDS niftkCore
  PACKAGE_DEPENDS OpenCV
)

target_link_libraries(niftkOpenCVUtils
  niftkcommon
  )


if (NIFTK_USE_COTIRE AND COMMAND cotire)
  cotire(niftkOpenCVUtils)
endif()

if(BUILD_TESTING)
  add_subdirectory(Testing)
endif()
