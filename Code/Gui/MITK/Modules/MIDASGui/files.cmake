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
  niftkBaseSegmentorControls.cxx
  niftkGeneralSegmentorControls.cxx
  niftkMorphologicalSegmentorControls.cxx
  niftkNewSegmentationDialog.cxx
  niftkSegmentationSelectorWidget.cxx
  niftkToolSelectorWidget.cxx
  niftkMIDASDrawToolGUI.cxx
  niftkMIDASPaintbrushToolGUI.cxx
)

set(MOC_H_FILES 
  niftkBaseSegmentorControls.h
  niftkGeneralSegmentorControls.h
  niftkMorphologicalSegmentorControls.h
  niftkNewSegmentationDialog.h
  niftkSegmentationSelectorWidget.h
  niftkToolSelectorWidget.h
  niftkMIDASDrawToolGUI.h
  niftkMIDASPaintbrushToolGUI.h
)

set(UI_FILES
  niftkGeneralSegmentorWidget.ui
  niftkMorphologicalSegmentorWidget.ui
  niftkSegmentationSelectorWidget.ui
  niftkToolSelectorWidget.ui
)
