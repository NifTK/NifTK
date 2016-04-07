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

# Tests with no extra command line parameter.
# Consider these as Developer-led, bare-bones unit tests.
set(MODULE_TESTS
  niftkPointBasedRegistrationTest.cxx
)

# For validation we prefer tests with externalised parameters.
set(MODULE_CUSTOM_TESTS
  niftkArunSVDRegTest.cxx
  niftkArunSVDExceptionTest.cxx
  niftkUltrasoundPointerBasedCalibrationTest.cxx
)
