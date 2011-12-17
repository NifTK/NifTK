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
#include <iostream>
#include <memory>
#include "ConversionUtils.h"
#include "itkImage.h"
#include "itkVector.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkTranslationTransform.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkNMIImageToImageMetric.h"
#include "itkNMILocalHistogramDerivativeForceFilter.h"
#include "itkFluidVelocityToDeformationFilter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "fftw3.h"

#define pi (3.14159265359)
#define tolerance (0.00001) 
#define Dimension (2)

/**
 * 
 */
class FluidVelocityToDeformationFilterUnitTest
{
public:
  /**
   * Typdefs.  
   */
  typedef vnl_vector<double> VnlVectorType;
  typedef vnl_matrix< double > VnlMatrixType; 
  typedef itk::FluidVelocityToDeformationFilter< double, Dimension > FluidVelocityToDeformationFilter;

  /**
   * 
   */
  int TestUpdateDeformationField()
  {
    FluidVelocityToDeformationFilter::Pointer fieldFilter = FluidVelocityToDeformationFilter::New();
    FluidVelocityToDeformationFilter::InputImageType::Pointer velocityField = FluidVelocityToDeformationFilter::InputImageType::New();
    FluidVelocityToDeformationFilter::InputImageType::Pointer deformationField = FluidVelocityToDeformationFilter::InputImageType::New();
    FluidVelocityToDeformationFilter::OutputImageType::Pointer outputField;
    
    typedef itk::ImageRegion< Dimension > RegionType;
    typedef itk::Size< Dimension > SizeType;
    SizeType size;
    RegionType region;
    int sizeX = 4;
    int sizeY = 4;
     
    size[0] = sizeX;
    size[1] = sizeY;
    region.SetSize(size);
    
    velocityField->SetRegions(region);
    velocityField->Allocate();

    deformationField->SetRegions(region);
    deformationField->Allocate();

    //[0, 0],[0, 0],[0, 0],[0, 0],
    //[0, 0],[1, 2],[2, 4],[0, 0],
    //[0, 0],[4, 4],[8, 8],[0, 0],
    //[0, 0],[0, 0],[0, 0],[0, 0],

    FluidVelocityToDeformationFilter::InputImageType::IndexType index;
    FluidVelocityToDeformationFilter::InputImageType::PixelType velocity; 
    FluidVelocityToDeformationFilter::InputImageType::PixelType deformation; 
    FluidVelocityToDeformationFilter::OutputImageType::IndexType outputIndex;
    FluidVelocityToDeformationFilter::OutputImageType::PixelType result;
    
    
    for (int x = 0; x < sizeX; x++)
    {
      for (int y = 0; y < sizeY; y++)
      {
        
        index[0] = x;
        index[1] = y;
        
        velocity[0] = x*x*y;
        velocity[1] = 2*x*y;
        
        velocityField->SetPixel(index, velocity);

        std::cerr << "velocity:" << index << "," << velocity << std::endl;
      }
    }
    
    for (int x = 0; x < sizeX; x++)
    {
      for (int y = 0; y < sizeY; y++)
      {
        
        index[0] = x;
        index[1] = y;
        deformation[0] = 2*x*y;
        deformation[1] = x*x*y;
        deformationField->SetPixel(index, deformation);
        
        std::cerr << "deformation:" << index << "," << deformation << std::endl;
        
      }
    }

    fieldFilter->SetCurrentDeformationField(deformationField);
    fieldFilter->SetVelocityField(velocityField);
    fieldFilter->Update();
    outputField = fieldFilter->GetOutput();
    
    for (int x = 0; x < sizeX; x++)
    {
      for (int y = 0; y < sizeY; y++)
      {
        outputIndex[0] = x;
        outputIndex[1] = y;
        result = outputField->GetPixel(outputIndex);
        
        std::cerr << "result:" << outputIndex << "," << result << std::endl;
        
        
      }
    }

    outputIndex[0] = 1;
    outputIndex[1] = 1;
    result = outputField->GetPixel(outputIndex);
    if (fabs(result[0] - (-5)) > tolerance) return EXIT_FAILURE;
    if (fabs(result[1] - (-2)) > tolerance) return EXIT_FAILURE;
    
    outputIndex[0] = 1;
    outputIndex[1] = 2;
    result = outputField->GetPixel(outputIndex);
    if (fabs(result[0] - (-14)) > tolerance) return EXIT_FAILURE;
    if (fabs(result[1] - (-8)) > tolerance) return EXIT_FAILURE;
    
    outputIndex[0] = 2;
    outputIndex[1] = 1;
    result = outputField->GetPixel(outputIndex);
    if (fabs(result[0] - (-20)) > tolerance) return EXIT_FAILURE;
    if (fabs(result[1] - (-28)) > tolerance) return EXIT_FAILURE;
    
    return EXIT_SUCCESS;
  }
};


int FluidVelocityToDeformationFilterTest(int argc, char * argv[])
{
  srand(time(NULL));
  
  FluidVelocityToDeformationFilterUnitTest test;
  
  if (test.TestUpdateDeformationField() != EXIT_SUCCESS)
    return EXIT_FAILURE;

  // All objects should be automatically destroyed at this point
  std::cout << "Test PASSED !" << std::endl;

  return EXIT_SUCCESS;

}
