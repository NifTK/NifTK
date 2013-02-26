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

SET(niftknvapi_SRCS
  niftkNVAPICommon.cpp
)

ADD_LIBRARY(niftknvapi ${niftknvapi_SRCS})
IF(BUILD_SHARED_LIBS)
  IF(WIN32)
    ADD_DEFINITIONS(-DNIFTKNVAPI_WINDOWS_EXPORT)
  ENDIF(WIN32)
ENDIF(BUILD_SHARED_LIBS)

TARGET_LINK_LIBRARIES(niftknvapi  
  ${NVAPI_LIBRARY}
)

MITK_INSTALL(TARGETS niftknvapi)
