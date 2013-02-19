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

SET(BASELINE ${NIFTK_DATA_DIR}/Baseline)
SET(INPUT_DATA ${NIFTK_DATA_DIR}/Input)

MAKE_DIRECTORY(${CMAKE_BINARY_DIR}/Testing/Temporary)
SET(TEMP ${CMAKE_BINARY_DIR}/Testing/Temporary)

ADD_SUBDIRECTORY(Code)
