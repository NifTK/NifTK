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
#  Last Changed      : $LastChangedDate: 2011-07-07 14:14:59 +0100 (Thu, 07 Jul 2011) $ 
#  Revision          : $Revision: 6684 $
#  Last modified by  : $Author: be $
#
#  Original author   : leung@drc.ion.ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/

# Go into sub-directories.
IF(BUILD_PROTOTYPE_JHH)
  SUBDIRS(jhh)
ENDIF(BUILD_PROTOTYPE_JHH)

IF(BUILD_PROTOTYPE_TM)
  SUBDIRS(tm)
ENDIF(BUILD_PROTOTYPE_TM)

IF(BUILD_PROTOTYPE_GY)
  SUBDIRS(gy)
ENDIF(BUILD_PROTOTYPE_GY)

IF(BUILD_PROTOTYPE_KKL)
  SUBDIRS(kkl)
ENDIF(BUILD_PROTOTYPE_KKL)

IF(BUILD_PROTOTYPE_BE)
  SUBDIRS(be)
ENDIF(BUILD_PROTOTYPE_BE)

IF(BUILD_PROTOTYPE_MJC)
  IF(VTK_FOUND)
    SUBDIRS(mjc)
  ENDIF(VTK_FOUND)
ENDIF(BUILD_PROTOTYPE_MJC)

IF(CUDA_FOUND)
  #ADD_SUBDIRECTORY(CUDA)
ENDIF(CUDA_FOUND)

IF(BUILD_COMMAND_LINE_PROGRAMS)
  SUBDIRS(InsightJournal)
ENDIF(BUILD_COMMAND_LINE_PROGRAMS)

