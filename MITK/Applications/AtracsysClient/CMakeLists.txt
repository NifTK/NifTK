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

if(Atracsys_FOUND)
  include_directories(${Atracsys_INCLUDE_DIR})
  NIFTK_CREATE_COMMAND_LINE_APPLICATION(
    NAME niftkAtracsysClient
    BUILD_SLICER
    INSTALL_SCRIPT
    TARGET_LIBRARIES
    niftkCore
    niftkAtracsys
    niftkOpenCVUtils
    ${Atracsys_LIBRARY}
  )
endif()

