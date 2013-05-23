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

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/anonymise.sh ${EXECUTABLE_OUTPUT_PATH}/anonymise.sh @ONLY )
MITK_INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/anonymise.sh)
