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

MITK_CREATE_MODULE_TESTS(LABELS niftkICPReg)

mitk_use_modules(TARGET ${TESTDRIVER}
  MODULES niftkVTK
)

mitkAddCustomModuleTest(SLS-mitkICPRegistrationTest_pointset niftkICPBasedRegistrationTest ${NIFTK_DATA_DIR}/Input/CylinderAxes_points.mps ${NIFTK_DATA_DIR}/Input/CylinderAxes.vtp)
mitkAddCustomModuleTest(SLS-mitkICPRegistrationTest_surface niftkICPBasedRegistrationTest ${NIFTK_DATA_DIR}/Input/CylinderAxes_points.vtk ${NIFTK_DATA_DIR}/Input/CylinderAxes.vtp)
mitkAddCustomModuleTest(SLS-mitkICPRegistrationTest_realdata niftkICPBasedRegistrationTestRealData ${NIFTK_DATA_DIR}/Input/stereo_matching_output.mps ${NIFTK_DATA_DIR}/Input/small_liver_phantom_to_points.vtk 20 200)

