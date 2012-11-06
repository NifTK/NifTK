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
#  Last Changed      : $LastChangedDate: 2011-10-19 12:16:53 +0100 (Thu, 07 Apr 2011) $ 
#  Revision          : $Revision: 1 $
#  Last modified by  : $Author: fp $
#
#  Original author   : f.carrasco@ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/

#--------------------------
# Install the shell scripts
#--------------------------
IF(NOT WIN32)

  CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/niftkBETter.sh.in ${SLICER_EXECUTABLE_OUTPUTPATH}/niftkBETter.sh @ONLY )
  INSTALL(PROGRAMS ${SLICER_EXECUTABLE_OUTPUTPATH}/niftkBETter.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR}/cli-modules COMPONENT scripts)
  
  CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/niftkNormalization.sh.in ${SLICER_EXECUTABLE_OUTPUTPATH}/niftkNormalization.sh @ONLY )
  INSTALL(PROGRAMS ${SLICER_EXECUTABLE_OUTPUTPATH}/niftkNormalization.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR}/cli-modules COMPONENT scripts)
  
ENDIF()