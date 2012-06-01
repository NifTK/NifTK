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
#  Last Changed      : $LastChangedDate: 2011-12-13 11:24:37 +0000 (Tue, 13 Dec 2011) $ 
#  Revision          : $Revision: 8004 $
#  Last modified by  : $Author: kkl $
#
#  Original author   : leung@drc.ion.ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/

# Go into sub-directories.
SUBDIRS(itkGroupwiseRegistration)
#SUBDIRS(Powerwatershed)
SUBDIRS(STAPLE)
IF(VTK_FOUND)
    SUBDIRS(ImportVtk)
ENDIF(VTK_FOUND)

##############################################################
# We only build niftkFluid if we have found FFTW. So add FFTW.
##############################################################
IF(FFTW_FOUND)
SUBDIRS(Fluid)
ENDIF(FFTW_FOUND)

#SUBDIRS(TBM)


