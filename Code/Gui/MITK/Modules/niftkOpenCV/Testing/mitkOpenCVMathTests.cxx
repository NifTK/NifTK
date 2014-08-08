/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#include <niftkFileHelper.h>
#include <mitkTestingMacros.h>
#include <mitkLogMacros.h>
#include <mitkOpenCVMaths.h>
#include <cmath>

/**
 * \file Tests for some of the functions in openCVMaths.
 */


bool RMSTest()
{
  // always start with this!
  MITK_TEST_BEGIN("mitkOpenCVMathRMSTest");

  //make some measured points
  std::vector < mitk::ProjectedPointPairsWithTimingError > measured;
  std::vector < mitk::ProjectedPointPairsWithTimingError > actual

  MITK_TEST_CONDITION ( true == true, "Testing the truth" );
  MITK_TEST_END();
}


int mitkOpenCVMathTests(int argc, char * argv[])
{
  // always start with this!
  MITK_TEST_BEGIN("mitkOpenCVMathTests");

  MITK_TEST_CONDITION ( true == true, "Testing the truth" );
  MITK_TEST_END();
}



