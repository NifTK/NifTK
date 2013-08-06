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

add_subdirectory(Exceptions)

if (BUILD_MESHING)
  add_subdirectory(Meshing)
endif (BUILD_MESHING)

set(niftkcommon_SRCS
FileHelper.cxx
ConversionUtils.cxx 
EnvironmentHelper.cxx
CommandLineParser.cxx
)

add_library(niftkcommon ${niftkcommon_SRCS})

if(BUILD_SHARED_LIBS)
  if(WIN32)
    add_definitions(-DNIFTKCOMMON_WINDOWS_EXPORT)
  endif(WIN32)
endif(BUILD_SHARED_LIBS)

target_link_libraries(niftkcommon  
  ITKCommon  
  ITKIO  
  ITKBasicFilters 
  ITKAlgorithms
  vtkCommon
  vtkIO
  ${Boost_LIBRARIES}   
) 

MITK_INSTALL(TARGETS niftkcommon)
