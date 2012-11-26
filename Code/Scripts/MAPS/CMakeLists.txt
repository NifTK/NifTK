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

IF(NOT WIN32)
  SET(scripts
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
  FOREACH(script ${scripts})
    CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/${script} ${EXECUTABLE_OUTPUT_PATH}/${script} @ONLY )
    MITK_INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/${script})
  ENDFOREACH()
ENDIF()



