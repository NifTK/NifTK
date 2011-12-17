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
#  Last Changed      : $LastChangedDate: 2010-05-26 10:55:12 +0100 (Wed, 26 May 2010) $ 
#  Revision          : $Revision: 3302 $
#  Last modified by  : $Author: mjc $
#
#  Original author   : m.clarkson@ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/

SET(BASELINE ${CMAKE_SOURCE_DIR}/Testing/Data/Baseline)
SET(INPUT_DATA ${CMAKE_SOURCE_DIR}/Testing/Data/Input)

MAKE_DIRECTORY(${CMAKE_BINARY_DIR}/Testing/Temporary)
SET(TEMP ${CMAKE_BINARY_DIR}/Testing/Temporary)

ADD_SUBDIRECTORY(Code)
