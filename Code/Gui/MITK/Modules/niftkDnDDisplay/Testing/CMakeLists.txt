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

set(QT_USE_QTTEST 1)
include(${QT_USE_FILE})

MITK_CREATE_MODULE_TESTS(LABELS niftkDnDDisplay)

# Trac 3146
mitkAddCustomModuleTest(DnDDisplay-SingleViewerWidgetTest niftkSingleViewerWidgetTest ${NIFTK_DATA_DIR}/Input/volunteers/16856/16856-002-1.img)
mitkAddCustomModuleTest(DnDDisplay-MultiViewerWidgetTest niftkMultiViewerWidgetTest ${NIFTK_DATA_DIR}/Input/volunteers/16856/16856-002-1.img)

target_link_libraries(niftkDnDDisplayTestDriver ${QT_QTTEST_LIBRARY})
