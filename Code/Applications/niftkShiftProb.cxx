/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "itkLogHelper.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

/*!
 * \file niftkShiftProb.cxx
 * \page niftkShiftProb
 * \section niftkShiftProbSummary Adjust prob. output from STAPLE. 
 *
 * This program adjusts prob. output from STAPLE. 
 * \li Dimensions: 4. 
 * \li Pixel type: Scalars only, of unsigned char, char, unsigned short, short, unsigned int, int, unsigned long, long, float. 
 *
 */

void Usage(char *exec)
{
  niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
  std::cout << "  " << std::endl;
  std::cout << "  This program adjusts prob. output from STAPLE" << std::endl;
  std::cout << "  " << std::endl;
  std::cout << "  " << exec << " inputFileName thresholdProb newProb outputFileName" << std::endl;
  std::cout << "  " << std::endl;
  std::cout << "*** [mandatory] ***" << std::endl << std::endl;
  std::cout << "    inputFileName        Input image " << std::endl;
  std::cout << "    thresholdProb        Threshold prob. " << std::endl;
  std::cout << "    newProb              New prob. " << std::endl;
  std::cout << "    outputFileName       Output image" << std::endl << std::endl;      
}


int main(int argc, char* argv[])
{
  
  // Parse command line args
  for(int i=1; i < argc; i++)
  {
    if (strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 || strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--h")==0)
    {
      Usage(argv[0]);
      return -1;
    }
  }
  
  if (argc < 5)
  {
    Usage(argv[0]); 
    return -1; 
  }
  
  const unsigned int Dimension = 4;
  typedef float PixelType;
  typedef itk::Image< PixelType, Dimension > FixedImageType;
  typedef itk::ImageFileReader< FixedImageType > FixedImageReaderType;
  typedef itk::ImageFileWriter< FixedImageType > FixedImageWriterType;
  
  // Read in the fixed images. 
  FixedImageReaderType::Pointer imageReader = FixedImageReaderType::New();
  FixedImageWriterType::Pointer writer = FixedImageWriterType::New(); 

  imageReader->SetFileName(argv[1]); 
  imageReader->Update(); 
  float confidence = atof(argv[2]); 
  float newConfidence = atof(argv[3]); 
  writer->SetFileName(argv[4]); 
  
  std::cout << "Threshlding at " << confidence << std::endl; 
  FixedImageType::RegionType::SizeType size = imageReader->GetOutput()->GetLargestPossibleRegion().GetSize(); 
  std::cout << "Image size = " << size << std::endl; 
  
  for (unsigned int x = 0; x < size[0]; x++)
  {
    for (unsigned int y = 0; y < size[1]; y++)
    {
      for (unsigned int z = 0; z < size[2]; z++)
      {
        FixedImageType::IndexType index; 
        
        index[0] = x; 
        index[1] = y; 
        index[2] = z;
        index[3] = 1;  
        
        PixelType value = imageReader->GetOutput()->GetPixel(index); 
        
        // Threshold the prob. 
        if (value < confidence)
        {
          value = std::min(value, newConfidence); 
          imageReader->GetOutput()->SetPixel(index, value); 
          
          FixedImageType::IndexType backgroundIndex = index;
          backgroundIndex[3] = 0; 
          imageReader->GetOutput()->SetPixel(backgroundIndex, 1.0-value); 
        }
      }
    }
  }

  writer->SetInput(imageReader->GetOutput()); 
  writer->Update(); 
  
  return 0;   
}
