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
#  Copyright (c) UCL : See LICENSE.txt in the top level directory for details..
#
#  Last Changed      : $LastChangedDate: 2011-12-16 09:02:17 +0000 (Fri, 16 Dec 2011) $.
#  Revision          : $Revision: 8038 $
#  Last modified by  : $Author: mjc $
#
#  Original author   : m.espak@cs.ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/

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
