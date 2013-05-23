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

if(NOT WIN32)
  set(scripts
    Hippo-MAPS-batch.sh
    Hippo-MAPS.sh
    MAPS-common.sh
    #Hippo-Propagate.sh
    #Hippo-Propagate-Batch.sh
    MAPS-build-template-library.sh
    MAPS-build-template-library-batch.sh
    #Generic-MAPS-batch.sh
    #Generic-MAPS.sh
    MAPS-build-brain-template-library.sh
    MAPS-build-brain-template-library-batch.sh
    Brain-MAPS-batch.sh
    Brain-MAPS.sh
    MAPS-brain-to-brain-registration-without-repeat-mask.sh
    #MAPS_groupwise_brain_registration.sh
  )
  foreach(script ${scripts})
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${script} ${EXECUTABLE_OUTPUT_PATH}/${script} @ONLY )
    MITK_INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/${script})
  endforeach()
endif()



