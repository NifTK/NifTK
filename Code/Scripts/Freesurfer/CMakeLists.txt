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
CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/fsImport.sh.in ${EXECUTABLE_OUTPUT_PATH}/fsImport.sh COPYONLY )
INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/fsImport.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT scripts)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/_fsInclude.sh.in ${EXECUTABLE_OUTPUT_PATH}/_fsInclude.sh COPYONLY )
INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/_fsInclude.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT scripts)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/_fsRunCleanup.sh.in ${EXECUTABLE_OUTPUT_PATH}/_fsRunCleanup.sh COPYONLY )
INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/_fsRunCleanup.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT scripts)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/fsRunCleanups.sh.in ${EXECUTABLE_OUTPUT_PATH}/fsRunCleanups.sh COPYONLY )
INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/fsRunCleanups.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT scripts)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/fsRunCrossSectionals.sh.in ${EXECUTABLE_OUTPUT_PATH}/fsRunCrossSectionals.sh COPYONLY )
INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/fsRunCrossSectionals.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT scripts)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/_fsRunCrossSectional.sh.in ${EXECUTABLE_OUTPUT_PATH}/_fsRunCrossSectional.sh COPYONLY )
INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/_fsRunCrossSectional.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT scripts)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/fsRunStats.sh.in ${EXECUTABLE_OUTPUT_PATH}/fsRunStats.sh COPYONLY )
INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/fsRunStats.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT scripts)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/fsTransferSubjects.sh.in ${EXECUTABLE_OUTPUT_PATH}/fsTransferSubjects.sh COPYONLY )
INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/fsTransferSubjects.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT scripts)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/fsRerunAfterEdits.sh.in ${EXECUTABLE_OUTPUT_PATH}/fsRerunAfterEdits.sh COPYONLY )
INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/fsRerunAfterEdits.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT scripts)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/_fsRerunAfterEdits.sh.in ${EXECUTABLE_OUTPUT_PATH}/_fsRerunAfterEdits.sh COPYONLY )
INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/_fsRerunAfterEdits.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT scripts)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/fsCheckJob.sh.in ${EXECUTABLE_OUTPUT_PATH}/fsCheckJob.sh COPYONLY )
INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/fsCheckJob.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT scripts)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/fsVoxelBasedGenerateThickness.sh.in ${EXECUTABLE_OUTPUT_PATH}/fsVoxelBasedGenerateThickness.sh COPYONLY )
INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/fsVoxelBasedGenerateThickness.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT scripts)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/_fsVoxelBasedGenerateThickness.sh.in ${EXECUTABLE_OUTPUT_PATH}/_fsVoxelBasedGenerateThickness.sh COPYONLY )
INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/_fsVoxelBasedGenerateThickness.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT scripts)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/fsVoxelBasedImport.sh.in ${EXECUTABLE_OUTPUT_PATH}/fsVoxelBasedImport.sh COPYONLY )
INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/fsVoxelBasedImport.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT scripts)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/fsVoxelBasedSegmentationRunCrossSectionals.sh.in ${EXECUTABLE_OUTPUT_PATH}/fsVoxelBasedSegmentationRunCrossSectionals.sh COPYONLY )
INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/fsVoxelBasedSegmentationRunCrossSectionals.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT scripts)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/_fsVoxelBasedSegmentationRunCrossSectional.sh.in ${EXECUTABLE_OUTPUT_PATH}/_fsVoxelBasedSegmentationRunCrossSectional.sh COPYONLY )
INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/_fsVoxelBasedSegmentationRunCrossSectional.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT scripts)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/fsImportUsingMRIConvert.sh.in ${EXECUTABLE_OUTPUT_PATH}/fsImportUsingMRIConvert.sh COPYONLY )
INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/fsImportUsingMRIConvert.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT scripts)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/fsImportVentricles.sh.in ${EXECUTABLE_OUTPUT_PATH}/fsImportVentricles.sh COPYONLY )
INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/fsImportVentricles.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT scripts)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/fsRunLongitudinalBaseTemplates.sh.in ${EXECUTABLE_OUTPUT_PATH}/fsRunLongitudinalBaseTemplates.sh COPYONLY )
INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/fsRunLongitudinalBaseTemplates.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT scripts)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/_fsRunLongitudinalBaseTemplate.sh.in ${EXECUTABLE_OUTPUT_PATH}/_fsRunLongitudinalBaseTemplate.sh COPYONLY )
INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/_fsRunLongitudinalBaseTemplate.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT scripts)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/fsRunLongitudinalAllTimepoints.sh.in ${EXECUTABLE_OUTPUT_PATH}/fsRunLongitudinalAllTimepoints.sh COPYONLY )
INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/fsRunLongitudinalAllTimepoints.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT scripts)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/_fsRunLongitudinalAllTimepoints.sh.in ${EXECUTABLE_OUTPUT_PATH}/_fsRunLongitudinalAllTimepoints.sh COPYONLY )
INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/_fsRunLongitudinalAllTimepoints.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT scripts)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/fsShowT1.sh.in ${EXECUTABLE_OUTPUT_PATH}/fsShowT1.sh COPYONLY )
INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/fsShowT1.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT scripts)
