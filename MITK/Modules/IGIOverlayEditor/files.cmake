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
  niftkBitmapOverlay.cxx
  niftkSingle3DViewWidget.cxx
  niftkSingleUltrasoundWidget.cxx
  niftkSingleVideoWidget.cxx
  niftkIGIVideoOverlayWidget.cxx
  niftkIGIUltrasoundOverlayWidget.cxx
)

set(MOC_H_FILES
  niftkSingle3DViewWidget.h
  niftkSingleUltrasoundWidget.h
  niftkSingleVideoWidget.h
  niftkIGIVideoOverlayWidget.h
  niftkIGIUltrasoundOverlayWidget.h
)

set(UI_FILES
  niftkIGIVideoOverlayWidget.ui
  niftkIGIUltrasoundOverlayWidget.ui
)

set(QRC_FILES
)

