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

SET(niftkVTK_SRCS
  Common/vtkFunctions.cxx
  niftkVTKIterativeClosestPoint.cxx
  niftkvtk4PointsReader.cxx
)

ADD_LIBRARY(niftkVTK ${niftkVTK_SRCS})

TARGET_LINK_LIBRARIES(niftkVTK
  niftkcommon
  ${NIFTK_VTK_LIBS_BUT_WITHOUT_QT}
  ${Boost_LIBRARIES}
  )

IF(BUILD_SHARED_LIBS)
  IF(WIN32)
    ADD_DEFINITIONS(-DNIFTKVTK_WINDOWS_EXPORT)
  ENDIF(WIN32)
ENDIF(BUILD_SHARED_LIBS)

MITK_INSTALL(TARGETS niftkVTK)
