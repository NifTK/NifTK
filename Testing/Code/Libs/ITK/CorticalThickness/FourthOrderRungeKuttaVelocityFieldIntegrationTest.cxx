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

#include "itkFourthOrderRungeKuttaVelocityFieldIntegrationFilter.h"

/**
 * Test the FourthOrderRungeKuttaVelocityFieldIntegrationFilter.
 */
int FourthOrderRungeKuttaVelocityFieldIntegrationTest(int, char* []) 
{
  
  typedef itk::FourthOrderRungeKuttaVelocityFieldIntegrationFilter<float, 2> FilterType;
  typedef FilterType::TimeVaryingVelocityImageType VelocityFieldType;
  typedef FilterType::TimeVaryingVelocityPixelType VelocityFieldPixelType;
  typedef FilterType::DisplacementPixelType DisplacementPixelType;
  typedef FilterType::MaskImageType MaskImageType;
  typedef FilterType::ThicknessImageType ThicknessImageType;
  
  VelocityFieldType::Pointer velocityField = VelocityFieldType::New();
  MaskImageType::Pointer maskImage = MaskImageType::New();
  ThicknessImageType::Pointer thicknessPriorImage = ThicknessImageType::New();
  
  // Create a velocity image, 5x5x2.
  VelocityFieldType::SizeType size;
  size[0] = 5;
  size[1] = 5;
  size[2] = 2;
  VelocityFieldType::IndexType index;
  index.Fill(0);
  VelocityFieldType::RegionType region;
  region.SetSize(size);
  region.SetIndex(index);
  velocityField->SetRegions(region);
  velocityField->Allocate();
  
  std::cout << "Created velocity field:\n" << velocityField->GetLargestPossibleRegion() \
    << "\nspacing=" <<  velocityField->GetSpacing() \
    << "\norigin=" << velocityField->GetOrigin() \
    << "\ndirection=\n" << velocityField->GetDirection() \
    << std::endl;
  
  VelocityFieldPixelType velocityPixel;
  
  for (unsigned int t = 0; t < 2; t++)
    {
      for (unsigned int y = 0; y < 5; y++)
        {
          for (unsigned int x = 0; x < 5; x++)
            {
              if (t == 0)
                {
                  velocityPixel.Fill(1);
                }
              else
                {
                  velocityPixel.Fill(2);
                }
              index[0] = x;
              index[1] = y;
              index[2] = t;
              velocityField->SetPixel(index, velocityPixel);
              
              std::cout << "Inserted velocity index=" << index << ", velocityPixel=" << velocityField->GetPixel(index) << std::endl;
            }
        }
    }
  
  MaskImageType::SizeType maskSize;
  maskSize[0] = 5;
  maskSize[1] = 5;
  MaskImageType::IndexType maskIndex;
  maskIndex.Fill(0);
  MaskImageType::RegionType maskRegion;
  maskRegion.SetSize(maskSize);
  maskRegion.SetIndex(maskIndex);
  maskImage->SetRegions(maskRegion);
  maskImage->Allocate();
  maskImage->FillBuffer(0);
  
  maskIndex[0] = 2;
  maskIndex[1] = 2;
  
  maskImage->SetPixel(maskIndex, 1);
  
  std::cout << "Inserted mask index=" << maskIndex << ", maskPixel=" << (int)(maskImage->GetPixel(maskIndex)) << std::endl;
  
  FilterType::Pointer filter = FilterType::New();
  filter->SetStartTime(0);
  filter->SetFinishTime(1);
  filter->SetDeltaTime(0.1);
  filter->SetGreyWhiteInterfaceMaskImage(maskImage);
  filter->SetVoxelsToIntegrateMaskImage(maskImage);
  filter->SetInput(velocityField);
  filter->SetCalculateThickness(true);
  filter->Update();
  
  DisplacementPixelType displacementPixel = filter->GetOutput()->GetPixel(maskIndex);
  float thicknessPixel = filter->GetCalculatedThicknessImage()->GetPixel(maskIndex);

  if (fabs(displacementPixel[0] - 1.49167) > 0.0001)
    {
      std::cout << "Expected 1.49167, but got:" << displacementPixel[0] << std::endl;
      return EXIT_FAILURE;
    }
  if (fabs(displacementPixel[1] - 1.49167) > 0.0001)
    {
      std::cout << "Expected 1.49167, but got:" << displacementPixel[1] << std::endl;
      return EXIT_FAILURE;
    }
  if (fabs(thicknessPixel - 2.10954) > 0.0001)
    {
      std::cout << "Expected 2.10954, but got:" << thicknessPixel << std::endl;
      return EXIT_FAILURE;
    }

  // Run it backwards.
  filter->SetStartTime(1);
  filter->SetFinishTime(0);
  filter->SetDeltaTime(0.1);
  filter->SetCalculateThickness(true);
  filter->Update();

  displacementPixel = filter->GetOutput()->GetPixel(maskIndex);
  if (fabs(displacementPixel[0] - -1.50833) > 0.0001)
    {
      std::cout << "Expected -1.50833, but got:" << displacementPixel[0] << std::endl;
      return EXIT_FAILURE;
    }
  if (fabs(displacementPixel[1] - -1.50833) > 0.0001)
    {
      std::cout << "Expected -1.50833, but got:" << displacementPixel[1] << std::endl;
      return EXIT_FAILURE;
    }

  thicknessPriorImage->SetRegions(maskRegion);
  thicknessPriorImage->Allocate();
  thicknessPriorImage->FillBuffer(1);
  
  // Run it with a thickness prior to stop integration.
  filter->SetStartTime(0);
  filter->SetFinishTime(1);
  filter->SetDeltaTime(0.1);
  filter->SetMaxDistanceMaskImage(thicknessPriorImage);
  filter->Update();

  displacementPixel = filter->GetOutput()->GetPixel(maskIndex);
  if (fabs(displacementPixel[0] - 0.775) > 0.0001)
    {
      std::cout << "Expected 0.775, but got:" << displacementPixel[0] << std::endl;
      return EXIT_FAILURE;
    }
  if (fabs(displacementPixel[1] - 0.775) > 0.0001)
    {
      std::cout << "Expected 0.775, but got:" << displacementPixel[1] << std::endl;
      return EXIT_FAILURE;
    }

  return EXIT_SUCCESS;
}
