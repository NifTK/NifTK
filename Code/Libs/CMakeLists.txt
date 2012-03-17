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
#  Last Changed      : $LastChangedDate: 2011-12-16 15:42:03 +0000 (Fri, 16 Dec 2011) $ 
#  Revision          : $Revision: 8057 $
#  Last modified by  : $Author: mjc $
#
#  Original author   : m.clarkson@ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/

ADD_SUBDIRECTORY(Common)
ADD_SUBDIRECTORY(MIDAS)
ADD_SUBDIRECTORY(ITK)

IF(VTK_FOUND)
  ADD_SUBDIRECTORY(VTK)
ENDIF(VTK_FOUND)

IF(QT_FOUND)
  ADD_SUBDIRECTORY(Qt)
ENDIF(QT_FOUND)
