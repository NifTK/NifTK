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
#include "itkLogHelper.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkConstantPadImageFilter.h"
#include "itkRegionOfInterestImageFilter.h"

/*!
 * \file niftkPadImage.cxx
 * \page niftkPadImage
 * \section niftkPadImageSummary Enlarge the image size and pad it with a background value.
 *
 * This program enlarges the image size and pad it with a background value. 
 * \li Dimensions: 3
 * \li Pixel type: Scalars only, of unsigned char, char, unsigned short, short, unsigned int, int. 
 *
 */

int main(int argc, char** argv)
{
  if (argc < 7) 
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cerr << std::endl;
    std::cout << "Program to enlarge the image size and pad it with a background value." << std::endl;
    std::cout << argv[0] << " input output size_x size_y size_z pad_value" << std::endl;
    exit(1); 
  }
  
  char* inputFilename = argv[1];
  char* outputFilename = argv[2];
  unsigned int imagePadSizeX = atoi(argv[3]);
  unsigned int imagePadSizeY = atoi(argv[4]);
  unsigned int imagePadSizeZ = atoi(argv[5]);
  int paddingValue = atoi(argv[6]);
  
  const unsigned int Dimension = 3;
  typedef int PixelType;
  typedef itk::Image< PixelType, Dimension > ImageType;
  
  typedef itk::ImageFileReader< ImageType  > ImageReaderType;
  ImageReaderType::Pointer imageReader = ImageReaderType::New();
  imageReader->SetFileName(inputFilename);
  imageReader->Update();
  
  typedef itk::ConstantPadImageFilter< ImageType, ImageType > PadImageFilterType; 
  PadImageFilterType::Pointer imagePaddingFilter = PadImageFilterType::New(); 
  ImageType::SizeType imageRegionSize = imageReader->GetOutput()->GetLargestPossibleRegion().GetSize();
  typedef itk::ImageFileWriter< ImageType >  WriterType;
  WriterType::Pointer writer = WriterType::New(); 
  
  std::cout << "imageRegionSize=" << imageRegionSize << std::endl; 
  std::cout << "imagePadSize=" << imagePadSizeX << "," << imagePadSizeY << "," << imagePadSizeZ << std::endl; 
     
  if (imagePadSizeX >= imageRegionSize[0] && imagePadSizeY >= imageRegionSize[1] && imagePadSizeZ >= imageRegionSize[2])
  {
    unsigned long lowerPadding[Dimension];
    lowerPadding[0] = (imagePadSizeX-imageRegionSize[0])/2; 
    lowerPadding[1] = (imagePadSizeY-imageRegionSize[1])/2; 
    lowerPadding[2] = (imagePadSizeZ-imageRegionSize[2])/2; 
    unsigned long upperPadding[Dimension];
    upperPadding[0] = imagePadSizeX-imageRegionSize[0]-lowerPadding[0]; 
    upperPadding[1] = imagePadSizeY-imageRegionSize[1]-lowerPadding[1]; 
    upperPadding[2] = imagePadSizeZ-imageRegionSize[2]-lowerPadding[2]; 
      
    std::cout << "lowerPadding=" << lowerPadding[0] << "," << lowerPadding[1] << "," << lowerPadding[2] << std::endl; 
    std::cout << "upperPadding=" << upperPadding[0] << "," << upperPadding[1] << "," << upperPadding[2] << std::endl; 
    imagePaddingFilter->SetInput(imageReader->GetOutput());
    imagePaddingFilter->SetPadLowerBound(lowerPadding);
    imagePaddingFilter->SetPadUpperBound(upperPadding);
    imagePaddingFilter->SetConstant(paddingValue);
    imagePaddingFilter->UpdateLargestPossibleRegion();
    imagePaddingFilter->Update();
    writer->SetInput(imagePaddingFilter->GetOutput());
  }
  else if (imagePadSizeX <= imageRegionSize[0] && imagePadSizeY <= imageRegionSize[1] && imagePadSizeZ <= imageRegionSize[2])
  {
    typedef itk::RegionOfInterestImageFilter< ImageType, ImageType > ROIImageFilterType; 
    ROIImageFilterType::Pointer imageExtractFilter = ROIImageFilterType::New(); 
    
    ImageType::RegionType region; 
    ImageType::SizeType cropSize; 
    cropSize[0] = imagePadSizeX; 
    cropSize[1] = imagePadSizeY; 
    cropSize[2] = imagePadSizeZ; 
    std::cout << "cropSize=" << cropSize << std::endl; 
    region.SetSize(cropSize); 
    ImageType::IndexType cropIndex; 
    cropIndex[0] = (imageRegionSize[0]-imagePadSizeX)/2; 
    cropIndex[1] = (imageRegionSize[1]-imagePadSizeY)/2; 
    cropIndex[2] = (imageRegionSize[2]-imagePadSizeZ)/2; 
    std::cout << "cropIndex=" << cropIndex << std::endl; 
    region.SetIndex(cropIndex); 
    
    imageExtractFilter->SetInput(imageReader->GetOutput());
    imageExtractFilter->SetRegionOfInterest(region); 
    imageExtractFilter->UpdateLargestPossibleRegion();
    imageExtractFilter->Update();
    writer->SetInput(imageExtractFilter->GetOutput());
  }
  
  writer->SetFileName(outputFilename);
  writer->Update();
  
  return 0; 
}
                                      

