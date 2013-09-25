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

# tests with no extra command line parameter
set(MODULE_TESTS
  mitkIGIDataSourceTest.cxx
  mitkTrackedImageCommandTest.cxx
  mitkTrackedPointerManagerTest.cxx
  mitkPointBasedRegistrationTest.cxx
  QDSCommonTest.cxx
)

set(MODULE_CUSTOM_TESTS
  mitkSurfaceBasedRegistrationTest.cxx
  mitkSurfaceBasedRegistrationTestRealData.cxx
)
