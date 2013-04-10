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
# Usage: niftkMacroGetChecksum(RESULT_VAR FILE_URI)
#
# Downloads the md5 checksum file for the file and stores the checksum
# in RESULT_VAR. It expects that the checksum file has the same name as
# the original file plus the '.md5' extension.
#

macro(niftkMacroGetChecksum RESULT_VAR FILE_URI)

  # We expect that the checksum has the name of the original file plus
  # the '.md5' extension.
  set(MD5_FILE_URI "${FILE_URI}.md5")

  # Cuts the host name and directory and keeps the file name only:
  string(REGEX REPLACE ".*/" "" MD5_FILE ${MD5_FILE_URI})

  # Downloads the md5 file:
  file(DOWNLOAD "${MD5_FILE_URI}" "${CMAKE_CURRENT_BINARY_DIR}/${MD5_FILE}")

  # Reads the first 32B to the output variable. (MD5 checksums are 128b.)
  file(STRINGS "${CMAKE_CURRENT_BINARY_DIR}/${MD5_FILE}" checksum LIMIT_INPUT 32)

  set(${RESULT_VAR} ${checksum})
endmacro(niftkMacroGetChecksum)
