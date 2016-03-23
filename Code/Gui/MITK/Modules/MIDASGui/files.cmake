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
  Dialogs/niftkMIDASNewSegmentationDialog.cxx
  niftkMIDASToolSelectorWidget.cxx
  niftkSegmentationSelectorWidget.cxx
  niftkMIDASDrawToolGUI.cxx
  niftkMIDASPaintbrushToolGUI.cxx
)

set(MOC_H_FILES 
  Dialogs/niftkMIDASNewSegmentationDialog.h
  niftkMIDASToolSelectorWidget.h
  niftkSegmentationSelectorWidget.h
  niftkMIDASDrawToolGUI.h
  niftkMIDASPaintbrushToolGUI.h
)

set(UI_FILES
  niftkSegmentationSelectorWidget.ui
  niftkMIDASToolSelector.ui
)
