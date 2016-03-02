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

NIFTK_CREATE_COMMAND_LINE_APPLICATION(
  NAME niftkUltrasoundPinCalibration
  BUILD_SLICER
  INSTALL_SCRIPT
  TARGET_LIBRARIES
    niftkcommon
    MitkLegacyIO
    ${Boost_LIBRARIES}
    MitkSegmentation
    niftkOpenCVUtils
    niftkIGIGui
    niftkPointReg
    niftkSurfRecon
    niftkICPReg
    niftkVTK
)
