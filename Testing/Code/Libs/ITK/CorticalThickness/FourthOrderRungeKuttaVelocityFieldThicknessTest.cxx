/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-05-28 18:04:05 +0100 (Fri, 28 May 2010) $
 Revision          : $Revision: 3325 $
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

#include "itkFourthOrderRungeKuttaVelocityFieldIntegrationFilter.h"
#include "itkImageFileWriter.h"

/**
 * Test the FourthOrderRungeKuttaVelocityFieldIntegrationFilter.
 */
int FourthOrderRungeKuttaVelocityFieldThicknessTest(int argc, char * argv[]) 
{

  if (argc < 2 )
    {
      std::cerr << "FourthOrderRungeKuttaVelocityFieldThicknessTest outputImage" << std::endl;
      return EXIT_FAILURE;
    }
  
  std::string outputImage = argv[1];
  std::cout << "Writing to:" << outputImage << std::endl;
  
  typedef itk::FourthOrderRungeKuttaVelocityFieldIntegrationFilter<float, 2> FilterType;
  typedef FilterType::TimeVaryingVelocityImageType VelocityFieldType;
  typedef FilterType::TimeVaryingVelocityPixelType VelocityFieldPixelType;
  typedef FilterType::DisplacementPixelType DisplacementPixelType;
  typedef FilterType::MaskImageType MaskImageType;
  typedef FilterType::ThicknessImageType ThicknessImageType;
  typedef itk::ImageFileWriter< ThicknessImageType > ThicknessImageWriterType;
  
  VelocityFieldType::Pointer velocityField = VelocityFieldType::New();
  MaskImageType::Pointer greyWhiteInterfaceImage = MaskImageType::New();
  ThicknessImageWriterType::Pointer writer = ThicknessImageWriterType::New();
  
  // Create a velocity image, 100x100x2
  VelocityFieldType::SizeType size;
  size[0] = 100;
  size[1] = 100;
  size[2] = 2;
  VelocityFieldType::IndexType index;
  index.Fill(0);
  VelocityFieldType::RegionType region;
  region.SetSize(size);
  region.SetIndex(index);
  velocityField->SetRegions(region);
  velocityField->Allocate();

  // Create a grey/white interface image, 100x100.
  MaskImageType::SizeType maskSize;
  maskSize[0] = 100;
  maskSize[1] = 100;
  MaskImageType::IndexType maskIndex;
  maskIndex.Fill(0);
  MaskImageType::RegionType maskRegion;
  maskRegion.SetSize(maskSize);
  maskRegion.SetIndex(maskIndex);
  greyWhiteInterfaceImage->SetRegions(maskRegion);
  greyWhiteInterfaceImage->Allocate();
  greyWhiteInterfaceImage->FillBuffer(0);

  FilterType::TimeVaryingVelocityPixelType velocityPixel;
  
  float middleX = 99/2.0;
  float middleY = middleX;

  for (int t = 0; t < 2; t++)
    {
      for (int x = 0; x < 100; x++)
        {
          for (int y = 0; y < 100; y++)
            {
                  index[0] = x;
                  index[1] = y;
                  index[2] = t;
                  
                  // Velocity field is going outwards, spherically from the middle.
                  float xdiff = x - middleX;
                  float ydiff = y - middleY;
                  
                  float length = sqrt(xdiff*xdiff + ydiff*ydiff);
                  
                  velocityPixel[0] = xdiff*2.0/length;
                  velocityPixel[1] = ydiff*2.0/length;

                  velocityField->SetPixel(index, velocityPixel);
            }
        }      
    }
  
  for (int x = 0; x < 100; x++)
    {
      for (int y = 0; y < 100; y++)
        {
          maskIndex[0] = x;
          maskIndex[1] = y;
          
          // We just want a single pixel wide ring, radius 25.
          
          float xdiff = x - middleX;
          float ydiff = y - middleY;

          float distance = sqrt(xdiff*xdiff + ydiff*ydiff);
          
          if (fabs(distance - 25) < 0.5)
            {
              greyWhiteInterfaceImage->SetPixel(maskIndex, 1);
            }
          else
            {
              greyWhiteInterfaceImage->SetPixel(maskIndex, 0);
            }
        }
    }
  
  std::cout << "Running filter" << std::endl;
  
  FilterType::Pointer filter = FilterType::New();
  filter->SetStartTime(0);
  filter->SetFinishTime(1);
  filter->SetDeltaTime(0.05);
  filter->SetInput(velocityField);
  filter->SetGreyWhiteInterfaceMaskImage(greyWhiteInterfaceImage);
  filter->SetCalculateThickness(true);
  filter->Update();

  writer->SetFileName(outputImage);
  writer->SetInput(filter->GetCalculatedThicknessImage());
  writer->Update();
  
  // Unit test compares image against a baseline.
  // Test the max thickness member variable output
  if (fabs(filter->GetMaxThickness() - 1.99997) > 0.0001)
  {
    std::cerr << "Expected 1.99997 but got " << filter->GetMaxThickness()  << std::endl;
    return EXIT_FAILURE;
  }
  
  return EXIT_SUCCESS;
}
