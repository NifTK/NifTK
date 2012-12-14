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
#  Last Changed      : $LastChangedDate: 2011-11-21 15:10:19 +0000 (Mon, 21 Nov 2011) $ 
#  Revision          : $Revision: 7831 $
#  Last modified by  : $Author: kkl $
#
#  Original author   : m.clarkson@ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/

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
