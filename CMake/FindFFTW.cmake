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

  
set(FFTW_FOUND 0)

set(NIFTK_FFTWINSTALL ${NIFTK_INSTALL_PREFIX})

message(${NIFTK_FFTWINSTALL})

find_path(FFTW_INCLUDE_DIR
  fftw3.h
  ${NIFTK_FFTWINSTALL}/include
  /usr/local/include
  /usr/include
  )

find_library(FFTW_LIBRARIES
  fftw3f       
  ${NIFTK_FFTWINSTALL}/lib
  /usr/local/lib
  /usr/lib
  )
  
find_library(FFTW_THREAD_LIBRARIES
  fftw3f_threads       
  ${NIFTK_FFTWINSTALL}/lib
  /usr/local/lib
  /usr/lib
  )
  
find_library(FFTW_OMP_LIBRARIES
  fftw3f_omp  
  ${NIFTK_FFTWINSTALL}/lib
  /usr/local/lib
  /usr/lib
  )


if(FFTW_LIBRARIES AND FFTW_INCLUDE_DIR)
  set(FFTW_FOUND 1)
endif(FFTW_LIBRARIES AND FFTW_INCLUDE_DIR)
