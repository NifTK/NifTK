/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <mitkTestingMacros.h>
#include <mitkLogMacros.h>

extern "C" {
#include <XnatRest.h>
}

void test_connect();

/** Documentation
 *  test for the XNAT REST C Library.
 */
int XnatRestTest(int /*argc*/, char* /*argv*/[])
{
  MITK_TEST_BEGIN("XnatRest");

  test_connect();

  // always end with this!
  MITK_TEST_END();
}

void test_connect() {

  MITK_TEST_CONDITION( true, "Testing XnatRest library");

  XnatRestStatus status;
  status = initXnatRest();

  MITK_TEST_CONDITION(status == XNATREST_OK, "Test connection");

  MITK_INFO << "first test succeeded" << std::endl;

/*
  MITK_TEST_CONDITION_REQUIRED( false,
                       "Testing the XnatRest library 4");
*/

}
