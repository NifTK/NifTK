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
#  Last Changed      : $LastChangedDate: 2010-07-28 12:37:56 +0100 (Wed, 28 Jul 2010) $ 
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
