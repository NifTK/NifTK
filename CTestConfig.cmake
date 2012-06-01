#/*================================================================================
#
#  NifTK: An image processing toolkit jointly developed by the
#              Dementia Research Centre, and the Centre For Medical Image Computing
#              at University College London.
#  
#  See:        http://dementia.ion.ucl.ac.uk/
#              http://cmic.cs.ucl.ac.uk/
#              http://www.ucl.ac.uk/
#
#  Copyright (c) UCL : See LICENSE.txt in the top level directory for details. 
#
#  Last Changed      : $LastChangedDate: 2010-05-25 16:52:43 +0100 (Tue, 25 May 2010) $ 
#  Revision          : $Revision: 3299 $
#  Last modified by  : $Author: mjc $
#
#  Original author   : m.clarkson@ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/

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


