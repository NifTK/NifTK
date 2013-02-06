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
#  Last Changed      : $LastChangedDate: 2011-05-25 18:13:39 +0100 (Wed, 25 May 2011) $ 
#  Revision          : $Revision: 6268 $
#  Last modified by  : $Author: kkl $
#
#  Original author   : m.clarkson@ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/
  
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
