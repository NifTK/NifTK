/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-12-13 10:54:10 +0000 (Tue, 13 Dec 2011) $
 Revision          : $Revision: 8003 $
 Last modified by  : $Author: kkl $

 Original author   : leung@drc.ion.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif


// this file defines the itkIOTests for the test driver
// and all it expects is that you have a function called RegisterTests
#include <iostream>
#include "itkTestMain.h" 

void RegisterTests()
{
  REGISTER_TEST(FluidDeformableTransformTest);
  REGISTER_TEST(FluidPDEFilterTest);
  REGISTER_TEST(FluidVelocityToDeformationFilterTest);
  REGISTER_TEST(itkFluidRegistrationTest);
  REGISTER_TEST(itkFluidRegistration3DTest);
  
}
