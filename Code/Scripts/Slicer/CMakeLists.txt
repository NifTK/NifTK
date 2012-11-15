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

SET(scripts)

IF(NOT WIN32)
  SET(scripts
    niftkBET.sh
    niftkCaminoMakeFAMDTRD.sh
    ${scripts}    
  )
ENDIF()

FOREACH(script ${scripts})
  CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/${script}.in ${SLICER_EXECUTABLE_OUTPUTPATH}/${script} @ONLY )
  INSTALL(PROGRAMS ${SLICER_EXECUTABLE_OUTPUTPATH}/${script} DESTINATION ${NIFTK_INSTALL_BIN_DIR}/cli-modules COMPONENT scripts)
ENDFOREACH()


