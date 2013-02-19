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

SET(CTEST_PROJECT_NAME "NifTK")
SET(CTEST_NIGHTLY_START_TIME "20:00:00 GMT")
SET(CTEST_TEST_TIMEOUT "3600")

IF(NOT DEFINED CTEST_DROP_METHOD)
  SET(CTEST_DROP_METHOD "http")
ENDIF(NOT DEFINED CTEST_DROP_METHOD)

IF(CTEST_DROP_METHOD STREQUAL "http")
  SET(CTEST_DROP_SITE "cmicdev.cs.ucl.ac.uk")
  SET(CTEST_DROP_LOCATION "/cdash/submit.php?project=NifTK")
  SET(CTEST_DROP_SITE_CDASH TRUE)
ENDIF(CTEST_DROP_METHOD STREQUAL "http")


