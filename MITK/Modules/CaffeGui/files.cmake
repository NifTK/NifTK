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

set(CPP_FILES
  Internal/niftkCaffeSegGUI.cxx
  niftkCaffeSegController.cxx
)

set(MOC_H_FILES 
  Internal/niftkCaffeSegGUI.h
  niftkCaffeSegController.h
)

set(UI_FILES
  Internal/niftkCaffeSegGUI.ui
)