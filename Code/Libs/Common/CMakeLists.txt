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

ADD_SUBDIRECTORY(Exceptions)

IF (BUILD_MESHING)
  ADD_SUBDIRECTORY(Meshing)
ENDIF (BUILD_MESHING)

SET(niftkcommon_SRCS
FileHelper.cxx
ConversionUtils.cxx 
EnvironmentHelper.cxx
CommandLineParser.cxx
)

ADD_LIBRARY(niftkcommon ${niftkcommon_SRCS})

IF(BUILD_SHARED_LIBS)
  IF(WIN32)
    ADD_DEFINITIONS(-DNIFTKCOMMON_WINDOWS_EXPORT)
  ENDIF(WIN32)
ENDIF(BUILD_SHARED_LIBS)

TARGET_LINK_LIBRARIES(niftkcommon  
  ITKCommon  
  ITKIO  
  ITKBasicFilters 
  ITKAlgorithms  
  vtkCommon
  vtkIO
  ${Boost_LIBRARIES}   
) 

MITK_INSTALL(TARGETS niftkcommon)
