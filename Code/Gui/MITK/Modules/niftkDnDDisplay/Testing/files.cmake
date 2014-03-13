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

set(MOC_H_FILES
  mitkQtSignalCollector.h
  niftkMultiViewerWidgetTest.h
  niftkSingleViewerWidgetTest.h
)

# tests with no extra command line parameter
set(MODULE_TESTS
)

set(MODULE_CUSTOM_TESTS
  niftkMultiViewerWidgetTest.cxx
  niftkSingleViewerWidgetTest.cxx
)
