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
  INCLUDE_DIRS Common CameraCalibration VideoProcessing Registration TagTracking demo
  DEPENDS mitkOpenCVVideoSupport MitkExt mitkCameraCalibration niftkCore
  PACKAGE_DEPENDS OpenCV
)

target_link_libraries(niftkOpenCV
  niftkcommon
  ${aruco_LIBS}
  )
    
add_subdirectory(Testing)
