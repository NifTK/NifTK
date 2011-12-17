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
#include <cfloat>
#include <math.h>
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkConstNeighborhoodIterator.h"


/**
 * Block-matching segmentation fusion. 
 */
int main(int argc, char * argv[])
{
  // Declare the types of the images
  const unsigned int Dimension = 3;
  typedef short PixelType;
  typedef itk::Image<PixelType, Dimension> ImageType;
  typedef itk::ImageFileReader<ImageType> ReaderType; 
  typedef itk::ImageFileWriter<ImageType> WriterType; 
  typedef itk::ConstNeighborhoodIterator<ImageType> IteratorType;
  
  const char* outputFileName = argv[1]; 
  const char* targetImageFileName = argv[2]; 
  const char* targetMaskFileName = argv[3]; 
  int blockRadius = atoi(argv[4]); 
  int searchRadius = atoi(argv[5]); 
  double power = atof(argv[6]); 
  
  int startingIndex = 7; 
      
  try
  {
    // Read in target image. 
    ReaderType::Pointer targetImageReader = ReaderType::New();
    targetImageReader->SetFileName(targetImageFileName); 
    targetImageReader->Update(); 
    
    // Read in target mask. 
    ReaderType::Pointer targetMaskReader = ReaderType::New();
    targetMaskReader->SetFileName(targetMaskFileName); 
    targetMaskReader->Update(); 
    
    // Read in images and segmentations to be fused.  
    int numberOfInputImage = (argc-startingIndex+1)/2; 
    ReaderType::Pointer* imageReader = new ReaderType::Pointer[numberOfInputImage]; 
    ReaderType::Pointer* labelReader = new ReaderType::Pointer[numberOfInputImage]; 
    for (int i = startingIndex; i < argc; i+=2)
    {
      int arrayIndex = (i-startingIndex)/2; 
      
      imageReader[arrayIndex] = ReaderType::New(); 
      imageReader[arrayIndex]->SetFileName(argv[i]); 
      imageReader[arrayIndex]->Update(); 
      
      labelReader[arrayIndex] = ReaderType::New(); 
      labelReader[arrayIndex]->SetFileName(argv[i+1]); 
      labelReader[arrayIndex]->Update(); 
      
      std::cout << "Input image file: " << argv[i] << std::endl; 
      std::cout << "Input label file: " << argv[i+1] << std::endl; 
    }
    
    // Allocate output image. 
    ReaderType::Pointer outputImageReader = ReaderType::New(); 
    outputImageReader->SetFileName(targetImageFileName); 
    outputImageReader->Update(); 
    outputImageReader->GetOutput()->FillBuffer(0); 
    
    IteratorType::RadiusType radius; 
    radius.Fill(blockRadius);
    
    // Mean squared distance. 
    double* bestMsd = new double[numberOfInputImage]; 
    // Mean absoluate distance. 
    double* bestMad = new double[numberOfInputImage]; 
    // Best label. 
    PixelType* bestLabel = new PixelType[numberOfInputImage]; 
    
    IteratorType targetImageIt(radius, targetImageReader->GetOutput(), targetImageReader->GetOutput()->GetLargestPossibleRegion());
    IteratorType targetMaskIt(radius, targetMaskReader->GetOutput(), targetMaskReader->GetOutput()->GetLargestPossibleRegion()); 
    for (targetImageIt.GoToBegin(), targetMaskIt.GoToBegin(); 
         !targetImageIt.IsAtEnd(); 
         ++targetImageIt, ++targetMaskIt)
    {
      
      if (targetMaskIt.GetCenterPixel() == 0)
        continue; 
      
      for (int j = 0; j < numberOfInputImage; j++)
      {
        bestMsd[j] = DBL_MAX; 
        bestMad[j] = DBL_MAX; 
        bestLabel[j] = 0; 
      }
      
      ImageType::IndexType targetImageIndex = targetImageIt.GetIndex(); 
      IteratorType::ConstIterator targetImageInnerIt; 
      IteratorType::ConstIterator atlasInnerIt; 
      
      // std::cout << "targetImageIndex=" << targetImageIndex << std::endl; 
      
      for (int j = 0; j < numberOfInputImage; j++)
      {
        // Serach for the best-matched block. 
        IteratorType atlasImageIt(radius, imageReader[j]->GetOutput(), imageReader[j]->GetOutput()->GetLargestPossibleRegion());
        ImageType::IndexType bestIndex; 
        
        for (int x = targetImageIndex[0]-searchRadius; x < targetImageIndex[0]+searchRadius; x++)
        {
          for (int y = targetImageIndex[1]-searchRadius; y < targetImageIndex[1]+searchRadius; y++)
          {
            for (int z = targetImageIndex[2]-searchRadius; z < targetImageIndex[2]+searchRadius; z++)
            {
              ImageType::IndexType atlasIndex; 
              atlasIndex[0] = x; 
              atlasIndex[1] = y; 
              atlasIndex[2] = z; 
              atlasImageIt.SetLocation(atlasIndex); 
              // Mean squared distance. 
              double msd = 0.0; 
              // Mean absoluate distance. 
              double mad = 0.0; 
              double count = 0.0; 
              
              for (targetImageInnerIt = targetImageIt.Begin(), atlasInnerIt = atlasImageIt.Begin(); 
                   targetImageInnerIt != targetImageIt.End(); 
                   ++targetImageInnerIt, ++atlasInnerIt)
              {
                // std::cout << "**targetImageInnerIt=" << **targetImageInnerIt << ", **atlasInnerIt=" << **atlasInnerIt << std::endl; 
                double diff = fabs((double)((**targetImageInnerIt) - (**atlasInnerIt))); 
                mad += diff;  
                msd += diff*diff; 
                count++; 
              }
              mad /= count; 
              msd /= count; 
              // std::cout << "mad=" << mad << std::endl; 
              
              if (mad < bestMad[j])
              {
                bestMad[j] = mad; 
              }
              if (msd < bestMsd[j])
              {
                bestMsd[j] = msd; 
                bestIndex = atlasIndex;
                bestLabel[j] = labelReader[j]->GetOutput()->GetPixel(atlasIndex); 
              }
            }
          }
        }
        // std::cout << "bestMad[0]=" << bestMad[0] << ", bestLabel[0]=" << bestLabel[0] << std::endl; 
        // std::cout << "bestIndex=" << bestIndex << std::endl; 
      }
      
      double weightedLabel = 0.0; 
      double totalWeight = 0.0; 
      for (int j = 0; j < numberOfInputImage; j++)
      {
        //weightedLabel += bestLabel[j]*pow(bestMad[j], power); 
        //totalWeight += pow(bestMad[j], power); 
        
        weightedLabel += bestLabel[j]*pow(bestMsd[j], power); 
        totalWeight += pow(bestMsd[j], power); 
        
        //weightedLabel += bestLabel[j]; 
        //totalWeight++; 
      }
      weightedLabel /= totalWeight; 
      
      // std::cout << "weightedLabel=" << weightedLabel << std::endl; 
          
      if (weightedLabel >= 0.5)
        outputImageReader->GetOutput()->SetPixel(targetImageIndex, 1); 
      else
        outputImageReader->GetOutput()->SetPixel(targetImageIndex, 0); 
    }
  
    WriterType::Pointer writer = WriterType::New(); 
    writer->SetInput(outputImageReader->GetOutput()); 
    writer->SetFileName(outputFileName); 
    writer->Update(); 
    
    
    delete[] imageReader; 
    delete[] labelReader; 
    delete[] bestMsd; 
    delete[] bestMad; 
  }
  catch( itk::ExceptionObject & err ) 
  { 
    std::cerr << "ExceptionObject caught !" << std::endl; 
    std::cerr << err << std::endl; 
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;    
}







