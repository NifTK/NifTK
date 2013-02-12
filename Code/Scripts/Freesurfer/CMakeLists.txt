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

IF(NOT WIN32)
  SET(scripts
    fsImport.sh
    _fsInclude.sh
    _fsRunCleanup.sh
    fsRunCleanups.sh
    fsRunCrossSectionals.sh
    _fsRunCrossSectional.sh
    fsRunStats.sh
    fsTransferSubjects.sh
    fsRerunAfterEdits.sh
    _fsRerunAfterEdits.sh
    fsCheckJob.sh
    fsVoxelBasedGenerateThickness.sh
    _fsVoxelBasedGenerateThickness.sh
    fsVoxelBasedImport.sh
    fsVoxelBasedSegmentationRunCrossSectionals.sh
    _fsVoxelBasedSegmentationRunCrossSectional.sh
    fsImportUsingMRIConvert.sh
    fsImportVentricles.sh
    fsRunLongitudinalBaseTemplates.sh
    _fsRunLongitudinalBaseTemplate.sh
    fsRunLongitudinalAllTimepoints.sh
    _fsRunLongitudinalAllTimepoints.sh
    fsShowT1.sh
  )
  FOREACH(script ${scripts})
    CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/${script}.in ${EXECUTABLE_OUTPUT_PATH}/${script} @ONLY )
    MITK_INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/${script})
  ENDFOREACH()
ENDIF()
