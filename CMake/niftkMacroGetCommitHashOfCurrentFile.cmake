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

#
# Usage: niftkMacroGetCommitHashOfCurrentFile(commit_hash_var)
#
# Retrieves the hash of the commit of the last modification of the CMake list file
# from which the macro is called.
# The macro stores the result in the 'commit_hash_var' variable.
#

macro(niftkMacroGetCommitHashOfCurrentFile commit_hash_var)

  execute_process(COMMAND ${GIT_EXECUTABLE} log -n 1 --pretty=format:%h -- ${CMAKE_CURRENT_LIST_FILE}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    ERROR_VARIABLE GIT_error
    OUTPUT_VARIABLE ${commit_hash_var}
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(NOT ${GIT_error} EQUAL 0)
    message(SEND_ERROR "Command \"${GIT_EXECUTBALE} log -n 1 --pretty=format:\"%h\" -- ${CMAKE_CURRENT_LIST_FILE} in directory ${CMAKE_SOURCE_DIR} failed with output:\n${GIT_error}")
  endif()

endmacro()
