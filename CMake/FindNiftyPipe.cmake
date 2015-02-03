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


set(NIFTYPIPE_FOUND 0)

if(NOT NIFTYPIPE_DIR)
  set(NIFTYPIPE_DIR ${NIFTK_LINK_PREFIX}/nifty_pipe CACHE PATH "Directory containing NiftyPipe installation")
else(NOT NIFTYPIPE_DIR)
  set(NIFTYPIPE_DIR @NIFTYPIPE_DIR@ CACHE PATH "Directory containing NiftyPipe installation")
endif(NOT NIFTYPIPE_DIR)
