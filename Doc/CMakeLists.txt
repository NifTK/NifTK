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

FILE(GLOB DOXIN_FILES *.dox.in)

FOREACH(doxin ${DOXIN_FILES})
  GET_FILENAME_COMPONENT(basename ${doxin} NAME_WE)
  CONFIGURE_FILE(${doxin} ${CMAKE_BINARY_DIR}/Doxygen/${basename}.dox)
ENDFOREACH()
