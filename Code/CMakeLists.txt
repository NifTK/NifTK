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
#  Last Changed      : $LastChangedDate: 2011-12-16 15:42:03 +0000 (Fri, 16 Dec 2011) $ 
#  Revision          : $Revision: 8057 $
#  Last modified by  : $Author: mjc $
#
#  Original author   : m.clarkson@ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/

ADD_SUBDIRECTORY(Libs)
IF(BUILD_GUI)
  ADD_SUBDIRECTORY(Gui)
ENDIF(BUILD_GUI)
IF(BUILD_COMMAND_LINE_PROGRAMS)
  ADD_SUBDIRECTORY(Applications)
ENDIF(BUILD_COMMAND_LINE_PROGRAMS)
ADD_SUBDIRECTORY(Scripts)

# Trac 1296. Turn these off for now

#FILE(GLOB_RECURSE H_FILES *.h *.txx)
#INSTALL(FILES ${H_FILES} DESTINATION ${NIFTK_INSTALL_INCLUDE_DIR} COMPONENT headers )





