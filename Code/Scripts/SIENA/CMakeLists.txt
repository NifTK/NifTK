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
#  Last Changed      : $LastChangedDate: 2011-04-07 12:16:53 +0100 (Thu, 07 Apr 2011) $ 
#  Revision          : $Revision: 5798 $
#  Last modified by  : $Author: mjc $
#
#  Original author   : m.clarkson@ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/

#--------------------------
# Install the shell scripts
#--------------------------
CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/_sienaNoBet.sh.in ${EXECUTABLE_OUTPUT_PATH}/_sienaNoBet.sh COPYONLY )
INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/_sienaNoBet.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT scripts)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/_sienaJob.sh.in ${EXECUTABLE_OUTPUT_PATH}/_sienaJob.sh COPYONLY )
INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/_sienaJob.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT scripts)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/sienaBatch.sh.in ${EXECUTABLE_OUTPUT_PATH}/sienaBatch.sh COPYONLY )
INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/sienaBatch.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT scripts)
