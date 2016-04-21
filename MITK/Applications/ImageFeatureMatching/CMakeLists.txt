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
  NAME niftkImageFeatureMatching
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
)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # Suppress warning because of a bug in the Boost gil library.
  set_target_properties(niftkImageFeatureMatching PROPERTIES COMPILE_FLAGS "-Wno-c++11-narrowing")
endif()
