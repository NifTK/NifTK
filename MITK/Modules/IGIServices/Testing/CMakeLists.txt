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

MITK_CREATE_MODULE_TESTS(LABELS niftkIGIServices)

mitk_use_modules(TARGET ${TESTDRIVER} PACKAGES Qt4|QtTest Qt5|Test)

mitkAddCustomModuleTest(REQ-2001-CSD-0001-008-01 niftkPointRegServiceRAIITest )
mitkAddCustomModuleTest(REQ-2001-CSD-0002-001-01 niftkICPRegServiceRAIITest ${NIFTK_DATA_DIR}/Input/stereo_matching_output.mps ${NIFTK_DATA_DIR}/Input/small_liver_phantom_to_points.vtk 20 200)
