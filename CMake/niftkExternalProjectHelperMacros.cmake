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


#########################################################################################
#
# Usage: niftkMacroGetCommitHashOfCurrentFile(commit_hash_var)
#
# Retrieves the hash of the commit of the last modification of the CMake list file
# from which the macro is called.
# The macro stores the result in the 'commit_hash_var' variable.
#
#########################################################################################


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


#########################################################################################
#
# Usage: niftkMacroDefineExternalProjectVariables(project_name version_number)
#
# Defines variables that are needed to set up an external project.
# The proj_DEPENDENCIES variable is set to an empty list. If the project depends
# on other external projects, it needs to be updated after the call of this macro.
#
#########################################################################################

macro(niftkMacroDefineExternalProjectVariables project version location)

  set(NIFTK_VERSION_${project} "${version}" CACHE STRING "Version of ${project}" FORCE)
  set(NIFTK_LOCATION_${project} "${location}" CACHE STRING "Location of ${project}" FORCE)
  mark_as_advanced(NIFTK_VERSION_${project})
  mark_as_advanced(NIFTK_LOCATION_${project})

  niftkMacroGetCommitHashOfCurrentFile(config_version)

  set(proj ${project})
  set(proj_VERSION ${version})
  set(proj_LOCATION ${location})
  set(proj_SOURCE ${EP_BASE}/${proj}-${proj_VERSION}-${config_version}-src)
  set(proj_CONFIG ${EP_BASE}/${proj}-${proj_VERSION}-${config_version}-cmake)
  set(proj_BUILD ${EP_BASE}/${proj}-${proj_VERSION}-${config_version}-build)
  set(proj_INSTALL ${EP_BASE}/${proj}-${proj_VERSION}-${config_version}-install)
  set(proj_DEPENDENCIES "")
  set(${project}_DEPENDS ${project})

endmacro()


#########################################################################################
#
# Usage: niftkMacroGetChecksum(RESULT_VAR FILE_URI)
#
# Downloads the md5 checksum file for the file and stores the checksum
# in RESULT_VAR. It expects that the checksum file has the same name as
# the original file plus the '.md5' extension.
#
#########################################################################################

macro(niftkMacroGetChecksum RESULT_VAR FILE_URI)

  # We expect that the checksum has the name of the original file plus
  # the '.md5' extension.
  set(MD5_FILE_URI "${FILE_URI}.md5")

  # Cuts the host name and directory and keeps the file name only:
  string(REGEX REPLACE ".*/" "" MD5_FILE ${MD5_FILE_URI})

  # Downloads the md5 file:
  file(DOWNLOAD "${MD5_FILE_URI}" "${proj_CONFIG}/src/${MD5_FILE}")

  # Reads the first 32B to the output variable. (MD5 checksums are 128b.)
  file(STRINGS "${proj_CONFIG}/src/${MD5_FILE}" checksum LIMIT_INPUT 32)

  set(${RESULT_VAR} ${checksum})
endmacro(niftkMacroGetChecksum)
