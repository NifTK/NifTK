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

SET(niftkMIDAS_SRCS
MidasCommon.cpp
MidasIO.cpp
)

ADD_LIBRARY(niftkmidas ${niftkMIDAS_SRCS})
IF(BUILD_SHARED_LIBS)
  IF(WIN32)
    ADD_DEFINITIONS(-DNIFTKMIDAS_WINDOWS_EXPORT)
  ENDIF(WIN32)
ENDIF(BUILD_SHARED_LIBS)

IF(BUILD_SHARED_LIBS)
  IF(WIN32)
    INSTALL(TARGETS niftkmidas 
      RUNTIME DESTINATION ${NIFTK_INSTALL_LIB_DIR} 
      COMPONENT libraries 
    )
    INSTALL(TARGETS niftkmidas 
      ARCHIVE DESTINATION ${NIFTK_INSTALL_LIB_DIR} 
      COMPONENT libraries 
    )    
  ELSE(WIN32)
    INSTALL(TARGETS niftkmidas 
      LIBRARY DESTINATION ${NIFTK_INSTALL_LIB_DIR} 
      ARCHIVE DESTINATION ${NIFTK_INSTALL_LIB_DIR} 
      COMPONENT libraries 
    )
  ENDIF(WIN32)
ELSE(BUILD_SHARED_LIBS)
  INSTALL(TARGETS niftkmidas 
    LIBRARY DESTINATION ${NIFTK_INSTALL_LIB_DIR} 
    ARCHIVE DESTINATION ${NIFTK_INSTALL_LIB_DIR} 
    COMPONENT libraries 
  )
ENDIF(BUILD_SHARED_LIBS)
