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
    _regInclude.sh
    regAIR.sh
    _regAIR.sh
    regADNIGenerateRandomlyScaledImages.sh
    regADNIGenerateRandomlyMisRegisteredImages.sh
    regADNIScaleImage.sh
    regAREG.sh
    regMisRegister.sh
    regADNIScaling.sh
    _regADNIScaling.sh
    _regAREG.sh
    symmetricReg.sh
    propogateRegion.sh
    propogateRegionBatch.sh
    propogationCheck.sh
    propogationCheckBatch.sh
    regFFD.sh
    regFFDBatch.sh
    _regAIR-standard-space.sh
    regAIR-standard-space.sh
    regAIRForVents.sh
    regAIRForVentsBatch.sh
    _regAIR-standard-space-all-timepoints.sh
    ComputePairwiseRegistration.sh
    ComputePairwiseRegistrationBatch.sh
    compute_symmetric_transform_batch.sh
    compute_symmetric_transform.sh
    compute_symmetric_transform_3.sh
    compute_symmetric_transform_n.sh
  )
  FOREACH(script ${scripts})
    CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/${script}.in ${EXECUTABLE_OUTPUT_PATH}/${script} @ONLY )
    MITK_INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/${script})
  ENDFOREACH()
ENDIF()

IF(NOT WIN32)
  SET(textfiles
    _regADNIScaling_brain_params.txt
    _regADNIScaling_skull_params.txt
  )
  FOREACH(textfile ${textfiles})
    CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/${textfile} ${EXECUTABLE_OUTPUT_PATH}/${textfile} @ONLY )
    MITK_INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/${textfile})
  ENDFOREACH()
ENDIF()
