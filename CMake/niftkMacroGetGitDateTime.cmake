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
#  Last Changed      : $LastChangedDate: $
#  Revision          : $Revision: $
#  Last modified by  : $Author: me $
#
#  Original author   : Miklos Espak <m.espak@ucl.ac.uk>
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/

macro(niftkMacroGetGitDateTime dir prefix)
  execute_process(COMMAND ${GIT_EXECUTABLE} show -s --format=%ci HEAD
       WORKING_DIRECTORY ${dir}
       ERROR_VARIABLE GIT_error
       OUTPUT_VARIABLE ${prefix}_DATE_TIME
       OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT ${GIT_error} EQUAL 0)
    message(SEND_ERROR "Command \"${GIT_EXECUTBALE} show -s --format=\"%ci\" HEAD\" in directory ${dir} failed with output:\n${GIT_error}")
  endif()
endmacro()
