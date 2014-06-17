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
#mitkAddCustomModuleTest(DnDDisplay-SingleViewerWidgetTest niftkSingleViewerWidgetTest ${NIFTK_DATA_DIR}/Input/volunteers/16856/16856-002-1.img)
mitkAddCustomModuleTest(DnDDisplay-MultiViewerWidgetTest niftkMultiViewerWidgetTest ${NIFTK_DATA_DIR}/Input/volunteers/16856/16856-002-1.img)

# Trac 3444
mitkAddCustomModuleTest(DnDDisplay-GeometryTest1 niftkSingleViewerWidgetTest ${NIFTK_DATA_DIR}/Input/volunteers/16856/16856-002-1.nii.gz testGetTimeGeometry)
mitkAddCustomModuleTest(DnDDisplay-GeometryTest2 niftkSingleViewerWidgetTest ${NIFTK_DATA_DIR}/Input/volunteers/16856/16856-002-1.nii.gz testSetTimeGeometry)
mitkAddCustomModuleTest(DnDDisplay-GeometryTest3 niftkSingleViewerWidgetTest ${NIFTK_DATA_DIR}/Input/volunteers/matt/mdeft_nifti/mdeft.1.nii testGetTimeGeometry)
mitkAddCustomModuleTest(DnDDisplay-GeometryTest4 niftkSingleViewerWidgetTest ${NIFTK_DATA_DIR}/Input/volunteers/matt/mdeft_nifti/mdeft.1.nii testSetTimeGeometry)
mitkAddCustomModuleTest(DnDDisplay-GeometryTest5 niftkSingleViewerWidgetTest ${NIFTK_DATA_DIR}/Input/volunteers/matt/mdeft_dicom/1.3.12.2.1107.5.2.32.35194.2009102113323066854212777.0.0.0_0002_000001_12561305650450.dcm testGetTimeGeometry)
mitkAddCustomModuleTest(DnDDisplay-GeometryTest6 niftkSingleViewerWidgetTest ${NIFTK_DATA_DIR}/Input/volunteers/matt/mdeft_dicom/1.3.12.2.1107.5.2.32.35194.2009102113323066854212777.0.0.0_0002_000001_12561305650450.dcm testSetTimeGeometry)

target_link_libraries(niftkDnDDisplayTestDriver ${QT_QTTEST_LIBRARY})
