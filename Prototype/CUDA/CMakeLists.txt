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
#  Last Changed      : $LastChangedDate: 2010-12-14 13:56:03 +0000 (Tue, 14 Dec 2010) $ 
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

SET(NAME cuda_utils)
SET(LIBRARY_NAME niftkcudautilsgpu)
CUDA_ADD_LIBRARY(${LIBRARY_NAME} ${NAME}_gpu.h ${NAME}_gpu.cu ${NAME}_kernels.cu)
INSTALL(TARGETS ${LIBRARY_NAME} 
  LIBRARY DESTINATION ${NIFTK_INSTALL_LIB_DIR} 
  ARCHIVE DESTINATION ${NIFTK_INSTALL_LIB_DIR}
  COMPONENT libraries 
)
