#/*================================================================================
#
#  NifTK: An image processing toolkit jointly developed by the
#              Dementia Research Centre, and the Centre For Medical Image Computing
#              at University College London.
#  
#  See:        http://dementia.ion.ucl.ac.uk/
#              http://cmic.cs.ucl.ac.uk/
#              http://www.ucl.ac.uk/
#
#  Copyright (c) UCL : See LICENSE.txt in the top level directory for details. 
#
#  Last Changed      : $LastChangedDate: 2011-11-30 18:35:54 +0000 (Wed, 30 Nov 2011) $ 
#  Revision          : $Revision: 7897 $
#  Last modified by  : $Author: gerge $
#
#  Original author   : m.clarkson@ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/

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
