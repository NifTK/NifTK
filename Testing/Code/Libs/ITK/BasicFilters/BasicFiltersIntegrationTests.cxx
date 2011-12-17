/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-10-06 10:55:39 +0100 (Thu, 06 Oct 2011) $
 Revision          : $Revision: 7447 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#include <iostream>
#include "itkTestMain.h" 

void RegisterTests()
{
  REGISTER_TEST(BoundaryValueRescaleIntensityImageFilterTest);
  REGISTER_TEST(SetOutputVectorToCurrentPositionFilterTest);
  REGISTER_TEST(VectorMagnitudeImageFilterTest);
  REGISTER_TEST(VectorVPlusLambdaUImageFilterTest);
  REGISTER_TEST(ShapeBasedAveragingImageFilterTest); 
  REGISTER_TEST(ExtractEdgeImageTest);
  REGISTER_TEST(MeanCurvatureImageFilterTest);
  REGISTER_TEST(GaussianCurvatureImageFilterTest);
}
