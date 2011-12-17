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
#  Original author   : leung@drc.ion.ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/Hippo-MAPS-batch.sh ${EXECUTABLE_OUTPUT_PATH}/Hippo-MAPS-batch.sh COPYONLY )
INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/Hippo-MAPS-batch.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT scripts)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/Hippo-MAPS.sh ${EXECUTABLE_OUTPUT_PATH}/Hippo-MAPS.sh COPYONLY )
INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/Hippo-MAPS.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT scripts)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/MAPS-common.sh ${EXECUTABLE_OUTPUT_PATH}/MAPS-common.sh COPYONLY )
INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/MAPS-common.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT scripts)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/Hippo-Propagate.sh ${EXECUTABLE_OUTPUT_PATH}/Hippo-Propagate.sh COPYONLY )
INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/Hippo-Propagate.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT scripts)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/Hippo-Propagate-Batch.sh ${EXECUTABLE_OUTPUT_PATH}/Hippo-Propagate-Batch.sh COPYONLY )
INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/Hippo-Propagate-Batch.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT scripts)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/MAPS-build-template-library.sh ${EXECUTABLE_OUTPUT_PATH}/MAPS-build-template-library.sh COPYONLY )
INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/MAPS-build-template-library.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT scripts)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/MAPS-build-template-library-batch.sh ${EXECUTABLE_OUTPUT_PATH}/MAPS-build-template-library-batch.sh COPYONLY )
INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/MAPS-build-template-library-batch.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT scripts)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/Generic-MAPS-batch.sh ${EXECUTABLE_OUTPUT_PATH}/Generic-MAPS-batch.sh COPYONLY )
INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/Generic-MAPS-batch.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT scripts)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/Generic-MAPS.sh ${EXECUTABLE_OUTPUT_PATH}/Generic-MAPS.sh COPYONLY )
INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/Generic-MAPS.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT scripts)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/MAPS-build-brain-template-library.sh ${EXECUTABLE_OUTPUT_PATH}/MAPS-build-brain-template-library.sh COPYONLY )
INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/MAPS-build-brain-template-library.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT scripts)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/MAPS-build-brain-template-library-batch.sh ${EXECUTABLE_OUTPUT_PATH}/MAPS-build-brain-template-library-batch.sh COPYONLY )
INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/MAPS-build-brain-template-library-batch.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT scripts)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/Brain-MAPS-batch.sh ${EXECUTABLE_OUTPUT_PATH}/Brain-MAPS-batch.sh COPYONLY )
INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/Brain-MAPS-batch.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT scripts)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/Brain-MAPS.sh ${EXECUTABLE_OUTPUT_PATH}/Brain-MAPS.sh COPYONLY )
INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/Brain-MAPS.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT scripts)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/MAPS-brain-to-brain-registration-without-repeat-mask.sh ${EXECUTABLE_OUTPUT_PATH}/MAPS-brain-to-brain-registration-without-repeat-mask.sh COPYONLY )
INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/MAPS-brain-to-brain-registration-without-repeat-mask.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT scripts)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/MAPS_groupwise_brain_registration.sh ${EXECUTABLE_OUTPUT_PATH}/MAPS_groupwise_brain_registration.sh COPYONLY )
INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/MAPS_groupwise_brain_registration.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT scripts)


