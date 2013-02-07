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

MITK_CREATE_MODULE(niftkOpenCV
  EXPORT_DEFINE NIFTKOPENCV_EXPORT
  INCLUDE_DIRS CameraCalibration demo
  PACKAGE_DEPENDS OpenCV
)

TARGET_LINK_LIBRARIES(niftkOpenCV
  niftkcommon
  )
    
ADD_SUBDIRECTORY(Testing)