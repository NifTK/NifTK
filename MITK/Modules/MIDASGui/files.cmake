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
  Internal/niftkBaseSegmentorGUI.cxx
  Internal/niftkDrawToolGUI.cxx
  Internal/niftkGeneralSegmentorEventInterface.cxx
  Internal/niftkGeneralSegmentorGUI.cxx
  Internal/niftkMorphologicalSegmentorGUI.cxx
  Internal/niftkNewSegmentationDialog.cxx
  Internal/niftkPaintbrushToolGUI.cxx
  Internal/niftkSegmentationSelectorWidget.cxx
  Internal/niftkToolSelectorWidget.cxx
  niftkBaseSegmentorController.cxx
  niftkGeneralSegmentorController.cxx
  niftkMorphologicalSegmentorController.cxx
)

set(MOC_H_FILES 
  Internal/niftkBaseSegmentorGUI.h
  Internal/niftkDrawToolGUI.h
  Internal/niftkGeneralSegmentorGUI.h
  Internal/niftkMorphologicalSegmentorGUI.h
  Internal/niftkNewSegmentationDialog.h
  Internal/niftkPaintbrushToolGUI.h
  Internal/niftkSegmentationSelectorWidget.h
  Internal/niftkToolSelectorWidget.h
  niftkBaseSegmentorController.h
  niftkGeneralSegmentorController.h
  niftkMorphologicalSegmentorController.h
)

set(UI_FILES
  Internal/niftkGeneralSegmentorWidget.ui
  Internal/niftkMorphologicalSegmentorWidget.ui
  Internal/niftkSegmentationSelectorWidget.ui
  Internal/niftkToolSelectorWidget.ui
)
