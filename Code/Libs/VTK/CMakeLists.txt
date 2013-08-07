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

set(niftkVTK_SRCS
  niftkVTKFunctions.cxx
  niftkVTKIterativeClosestPoint.cxx
  niftkVTK4PointsReader.cxx
)

add_library(niftkVTK ${niftkVTK_SRCS})

target_link_libraries(niftkVTK
  ${NIFTK_VTK_LIBS_BUT_WITHOUT_QT}
  )

if(BUILD_SHARED_LIBS)
  if(WIN32)
    add_definitions(-DNIFTKVTK_WINDOWS_EXPORT)
  endif(WIN32)
endif(BUILD_SHARED_LIBS)

MITK_INSTALL(TARGETS niftkVTK)
