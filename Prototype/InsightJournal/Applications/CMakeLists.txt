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
#  Last Changed      : $LastChangedDate$ 
#  Revision          : $Revision$
#  Last modified by  : $Author$
#
#  Original author   : m.clarkson@ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/
  
###############################################################
# This list of libraries serves as a default list for all apps.
###############################################################

SET(NIFTK_APPLICATION_LINK_LIBRARIES 
  niftkcommon 
  niftkITK
  ${NIFTK_ITK_LIB}
  ${Boost_LIBRARIES} 
)

ADD_EXECUTABLE(niftkSkeletonize niftkSkeletonize.cxx )
TARGET_LINK_LIBRARIES(niftkSkeletonize ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkSkeletonize RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
