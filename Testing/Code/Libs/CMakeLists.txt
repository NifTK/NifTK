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

ADD_SUBDIRECTORY(Common)
ADD_SUBDIRECTORY(MIDAS)
ADD_SUBDIRECTORY(ITK)

#IF(VTK_FOUND)
#  ADD_SUBDIRECTORY(VTK)
#ENDIF(VTK_FOUND)

