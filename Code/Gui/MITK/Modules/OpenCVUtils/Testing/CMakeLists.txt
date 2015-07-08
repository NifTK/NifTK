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

MITK_CREATE_MODULE_TESTS(LABELS niftkOpenCVUtils)

mitk_use_modules(TARGET ${TESTDRIVER} PACKAGES OpenCV)

mitkAddCustomModuleTest(mitkOpenCVFileIOUtilsTests mitkOpenCVFileIOUtilsTests ${NIFTK_DATA_DIR}/Input/UltrasoundPinCalibration/2015.04.23-ultrasound_calibration/)

