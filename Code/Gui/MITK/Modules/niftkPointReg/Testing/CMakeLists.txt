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

MITK_CREATE_MODULE_TESTS()

# For validation we prefer tests with externalised parameters.
# Keep each test definition on one line, as the documentation parsing process searches by line.
# Note that the test ID has to exactly match the requirement ID.

mitkAddCustomModuleTest(REQ-2001-CSD-0001-001-01 niftkArunSVDExceptionTest ${NIFTK_DATA_DIR}/Input-ISO/2001/2001-CSD-0001-001-01-fixed.mps ${NIFTK_DATA_DIR}/Input-ISO/2001/2001-CSD-0001-001-01-moving.mps)
mitkAddCustomModuleTest(REQ-2001-CSD-0001-002-01 niftkArunSVDExceptionTest ${NIFTK_DATA_DIR}/Input-ISO/2001/2001-CSD-0001-002-01-fixed.mps ${NIFTK_DATA_DIR}/Input-ISO/2001/2001-CSD-0001-002-01-moving.mps)
mitkAddCustomModuleTest(REQ-2001-CSD-0001-003-01 niftkArunSVDExceptionTest ${NIFTK_DATA_DIR}/Input-ISO/2001/2001-CSD-0001-003-01-fixed.mps ${NIFTK_DATA_DIR}/Input-ISO/2001/2001-CSD-0001-003-01-moving.mps)
mitkAddCustomModuleTest(REQ-2001-CSD-0001-004-01 niftkArunSVDRegTest ${NIFTK_DATA_DIR}/Input-ISO/2001/2001-CSD-0001-004-01-fixed.mps ${NIFTK_DATA_DIR}/Input-ISO/2001/2001-CSD-0001-004-01-moving.mps ${NIFTK_DATA_DIR}/Input-ISO/2001/2001-CSD-0001-004-01-expected.4x4 0.001)
mitkAddCustomModuleTest(REQ-2001-CSD-0001-005-01 niftkArunSVDRegTest ${NIFTK_DATA_DIR}/Input-ISO/2001/2001-CSD-0001-005-01-fixed.mps ${NIFTK_DATA_DIR}/Input-ISO/2001/2001-CSD-0001-005-01-moving.mps ${NIFTK_DATA_DIR}/Input-ISO/2001/2001-CSD-0001-005-01-expected.4x4 0.001)
mitkAddCustomModuleTest(Ultrasound-Pointer niftkUltrasoundPointerBasedCalibrationTest ${NIFTK_DATA_DIR}/Input/UltrasoundPointerCalibration/UltrasoundPointerCalibration-image.mps ${NIFTK_DATA_DIR}/Input/UltrasoundPointerCalibration/UltrasoundPointerCalibration-sensor.mps ${NIFTK_DATA_DIR}/Input/UltrasoundPointerCalibration/UltrasoundPointerCalibration-expected.4x4 1.0)

