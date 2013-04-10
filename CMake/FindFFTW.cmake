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

  
SET(FFTW_FOUND 0)

SET(NIFTK_FFTWINSTALL ${NIFTK_INSTALL_PREFIX})

MESSAGE(${NIFTK_FFTWINSTALL})

FIND_PATH(FFTW_INCLUDE_DIR
  fftw3.h
  ${NIFTK_FFTWINSTALL}/include
  /usr/local/include
  /usr/include
  )

FIND_LIBRARY(FFTW_LIBRARIES
  fftw3f       
  ${NIFTK_FFTWINSTALL}/lib
  /usr/local/lib
  /usr/lib
  )
  
FIND_LIBRARY(FFTW_THREAD_LIBRARIES
  fftw3f_threads       
  ${NIFTK_FFTWINSTALL}/lib
  /usr/local/lib
  /usr/lib
  )
  
FIND_LIBRARY(FFTW_OMP_LIBRARIES
  fftw3f_omp  
  ${NIFTK_FFTWINSTALL}/lib
  /usr/local/lib
  /usr/lib
  )


IF(FFTW_LIBRARIES AND FFTW_INCLUDE_DIR)
  SET(FFTW_FOUND 1)
ENDIF(FFTW_LIBRARIES AND FFTW_INCLUDE_DIR)
