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
 * \file Test harness for inversion routines.
 */
int mitkMatrixInvertTest(int argc, char * argv[])
{
  // always start with this!
  MITK_TEST_BEGIN("mitkMatrixInvertTest");

  // populate with data, which must be orthonormal.

  cv::Matx44d m1 = mitk::ConstructRigidTransformationMatrix(10, 20, 30, 40, 50, 60);
  cv::Matx44d m1InvertedUsingDecomposition = m1.inv();
  cv::Matx44d m1InvertedUsingTranspose;
  mitk::InvertRigid4x4Matrix(m1, m1InvertedUsingTranspose);

  for (unsigned int r = 0; r < 4; r++)
  {
    for (unsigned int c = 0; c < 4; c++)
    {
      MITK_TEST_CONDITION ( fabs(m1InvertedUsingDecomposition(r,c) -  m1InvertedUsingTranspose(r,c)) < 0.0001, "Testing r=" << r << ", c=" << c << ", d=" << m1InvertedUsingDecomposition(r,c) << ", t=" << m1InvertedUsingTranspose(r,c));
    }
  }
  MITK_TEST_END();
}


