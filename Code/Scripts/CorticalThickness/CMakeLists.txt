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
CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/cteOverlayThicknessImageOnFreeSurfer.sh.in ${EXECUTABLE_OUTPUT_PATH}/cteOverlayThicknessImageOnFreeSurfer.sh COPYONLY )
INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/cteOverlayThicknessImageOnFreeSurfer.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT scripts)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/cteRunCrossSectional.sh.in ${EXECUTABLE_OUTPUT_PATH}/cteRunCrossSectional.sh @ONLY )
INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/cteRunCrossSectional.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT scripts)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/cteRunCrossSectionalBatch.sh.in ${EXECUTABLE_OUTPUT_PATH}/cteRunCrossSectionalBatch.sh COPYONLY )
INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/cteRunCrossSectionalBatch.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT scripts)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/cteRunLongitudinal.sh.in ${EXECUTABLE_OUTPUT_PATH}/cteRunLongitudinal.sh COPYONLY )
INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/cteRunLongitudinal.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT scripts)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/cteRegisterAtlasToImage.sh.in ${EXECUTABLE_OUTPUT_PATH}/cteRegisterAtlasToImage.sh @ONLY )
INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/cteRegisterAtlasToImage.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT scripts)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/cteRegisterThicknessImageToAtlas.sh.in ${EXECUTABLE_OUTPUT_PATH}/cteRegisterThicknessImageToAtlas.sh @ONLY )
INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/cteRegisterThicknessImageToAtlas.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT scripts)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/cteRefineLabelImage.sh.in ${EXECUTABLE_OUTPUT_PATH}/cteRefineLabelImage.sh COPYONLY )
INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/cteRefineLabelImage.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT scripts)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/cteExtractStatsUsingAtlas.sh.in ${EXECUTABLE_OUTPUT_PATH}/cteExtractStatsUsingAtlas.sh COPYONLY )
INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/cteExtractStatsUsingAtlas.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT scripts)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/cteMaskImageUsingRegions.sh.in ${EXECUTABLE_OUTPUT_PATH}/cteMaskImageUsingRegions.sh COPYONLY )
INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/cteMaskImageUsingRegions.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT scripts)
 