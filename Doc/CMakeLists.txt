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

FILE(GLOB DOXIN_FILES *.dox.in)

FOREACH(doxin ${DOXIN_FILES})
  GET_FILENAME_COMPONENT(basename ${doxin} NAME_WE)
  CONFIGURE_FILE(${doxin} ${CMAKE_BINARY_DIR}/Doxygen/${basename}.dox @ONLY)
ENDFOREACH()
