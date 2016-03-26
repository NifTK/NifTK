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
  niftkBaseSegmentorGUI.cxx
  niftkDrawToolGUI.cxx
  niftkGeneralSegmentorGUI.cxx
  niftkMorphologicalSegmentorGUI.cxx
  niftkNewSegmentationDialog.cxx
  niftkPaintbrushToolGUI.cxx
  niftkSegmentationSelectorWidget.cxx
  niftkToolSelectorWidget.cxx
)

set(MOC_H_FILES 
  niftkBaseSegmentorGUI.h
  niftkDrawToolGUI.h
  niftkGeneralSegmentorGUI.h
  niftkMorphologicalSegmentorGUI.h
  niftkNewSegmentationDialog.h
  niftkPaintbrushToolGUI.h
  niftkSegmentationSelectorWidget.h
  niftkToolSelectorWidget.h
)

set(UI_FILES
  niftkGeneralSegmentorWidget.ui
  niftkMorphologicalSegmentorWidget.ui
  niftkSegmentationSelectorWidget.ui
  niftkToolSelectorWidget.ui
)
