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
#include <itkLogHelper.h>
#include <itkCommandLineHelper.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkConstantPadImageFilter.h>
#include <itkRegionOfInterestImageFilter.h>

/*!
 * \file niftkPadImage.cxx
 * \page niftkPadImage
 * \section niftkPadImageSummary Enlarge the image size and pad it with a background value.
 *
 * This program enlarges the image size and pad it with a background value. 
 * \li Pixel type: Scalars only, of unsigned char, char, unsigned short, short, unsigned int, int. 
 *
 */

struct arguments
{
  std::string inputFilename;
  std::string outputFilename;

  unsigned int imagePadSizeX;
  unsigned int imagePadSizeY;
  unsigned int imagePadSizeZ;

  int paddingValue;
};


template <int Dimension, class PixelType> 
int DoMain(arguments args)
{  

  typedef itk::Image< PixelType, Dimension > ImageType;

  std::cout << "Reading input image: " << args.inputFilename << std::endl;
  
  typedef itk::ImageFileReader< ImageType  > ImageReaderType;
  typename ImageReaderType::Pointer imageReader = ImageReaderType::New();
  imageReader->SetFileName(args.inputFilename);
  imageReader->Update();
  
  typedef itk::ConstantPadImageFilter< ImageType, ImageType > PadImageFilterType; 
  typename PadImageFilterType::Pointer imagePaddingFilter = PadImageFilterType::New(); 
  typename ImageType::SizeType imageRegionSize = imageReader->GetOutput()->GetLargestPossibleRegion().GetSize();
  typedef itk::ImageFileWriter< ImageType >  WriterType;
  typename WriterType::Pointer writer = WriterType::New(); 
  
  std::cout << "imageRegionSize=" << imageRegionSize << std::endl; 
  std::cout << "imagePadSize=" << args.imagePadSizeX << "," << args.imagePadSizeY << "," << args.imagePadSizeZ << std::endl; 
     
  if (args.imagePadSizeX >= imageRegionSize[0] && args.imagePadSizeY >= imageRegionSize[1] && args.imagePadSizeZ >= imageRegionSize[2])
  {
    unsigned long lowerPadding[Dimension];
    lowerPadding[0] = (args.imagePadSizeX-imageRegionSize[0])/2; 
    lowerPadding[1] = (args.imagePadSizeY-imageRegionSize[1])/2; 
    lowerPadding[2] = (args.imagePadSizeZ-imageRegionSize[2])/2; 
    unsigned long upperPadding[Dimension];
    upperPadding[0] = args.imagePadSizeX-imageRegionSize[0]-lowerPadding[0]; 
    upperPadding[1] = args.imagePadSizeY-imageRegionSize[1]-lowerPadding[1]; 
    upperPadding[2] = args.imagePadSizeZ-imageRegionSize[2]-lowerPadding[2]; 
      
    std::cout << "lowerPadding=" << lowerPadding[0] << "," << lowerPadding[1] << "," << lowerPadding[2] << std::endl; 
    std::cout << "upperPadding=" << upperPadding[0] << "," << upperPadding[1] << "," << upperPadding[2] << std::endl; 
    imagePaddingFilter->SetInput(imageReader->GetOutput());
    imagePaddingFilter->SetPadLowerBound(lowerPadding);
    imagePaddingFilter->SetPadUpperBound(upperPadding);
    imagePaddingFilter->SetConstant(args.paddingValue);
    imagePaddingFilter->UpdateLargestPossibleRegion();
    imagePaddingFilter->Update();
    writer->SetInput(imagePaddingFilter->GetOutput());
  }
  else if (args.imagePadSizeX <= imageRegionSize[0] && args.imagePadSizeY <= imageRegionSize[1] && args.imagePadSizeZ <= imageRegionSize[2])
  {
    typedef itk::RegionOfInterestImageFilter< ImageType, ImageType > ROIImageFilterType; 
    typename ROIImageFilterType::Pointer imageExtractFilter = ROIImageFilterType::New(); 
    
    typename ImageType::RegionType region; 
    typename ImageType::SizeType cropSize; 
    cropSize[0] = args.imagePadSizeX; 
    cropSize[1] = args.imagePadSizeY; 
    cropSize[2] = args.imagePadSizeZ; 
    std::cout << "cropSize=" << cropSize << std::endl; 
    region.SetSize(cropSize); 
    typename ImageType::IndexType cropIndex; 
    cropIndex[0] = (imageRegionSize[0]-args.imagePadSizeX)/2; 
    cropIndex[1] = (imageRegionSize[1]-args.imagePadSizeY)/2; 
    cropIndex[2] = (imageRegionSize[2]-args.imagePadSizeZ)/2; 
    std::cout << "cropIndex=" << cropIndex << std::endl; 
    region.SetIndex(cropIndex); 
    
    imageExtractFilter->SetInput(imageReader->GetOutput());
    imageExtractFilter->SetRegionOfInterest(region); 
    imageExtractFilter->UpdateLargestPossibleRegion();
    imageExtractFilter->Update();
    writer->SetInput(imageExtractFilter->GetOutput());
  }
  
  writer->SetFileName( args.outputFilename );
  writer->Update();

  return EXIT_SUCCESS;
}

/**
 * \brief Takes image1 and image2 and adds them together using itk::AddImageFilter.
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


  struct arguments args;

  args.inputFilename = argv[1];
  args.outputFilename = argv[2];

  args.imagePadSizeX = atoi(argv[3]);
  args.imagePadSizeY = atoi(argv[4]);
  args.imagePadSizeZ = atoi(argv[5]);

  args.paddingValue = atoi(argv[6]);
  
  

  // Validate command line args
  if (args.inputFilename.length() == 0 ||
      args.outputFilename.length() == 0)
  {
    return EXIT_FAILURE;
  }

  int dims = itk::PeekAtImageDimension(args.inputFilename);

  if (dims != 2 && dims != 3)
  {
    std::cout << "ERROR: Unsupported image dimension" << std::endl;
    return EXIT_FAILURE;
  }
  
  int result;

  switch (itk::PeekAtComponentType(args.inputFilename))
  {
  case itk::ImageIOBase::UCHAR:

    result = DoMain<3, unsigned char>(args);
    break;

  case itk::ImageIOBase::CHAR:

    result = DoMain<3, char>(args);
    break;

  case itk::ImageIOBase::USHORT:

    result = DoMain<3, unsigned short>(args);
    break;

  case itk::ImageIOBase::SHORT:

    result = DoMain<3, short>(args);
    break;

  case itk::ImageIOBase::UINT:

    result = DoMain<3, unsigned int>(args);
    break;

  case itk::ImageIOBase::INT:

    result = DoMain<3, int>(args);
    break;

  case itk::ImageIOBase::ULONG:

    result = DoMain<3, unsigned long>(args);
    break;

  case itk::ImageIOBase::LONG:

    result = DoMain<3, long>(args);
    break;

  case itk::ImageIOBase::FLOAT:

    result = DoMain<3, float>(args);
    break;

  case itk::ImageIOBase::DOUBLE:

    result = DoMain<3, double>(args);
    break;

  default:
    std::cerr << "ERROR: Non standard pixel format" << std::endl;
    return EXIT_FAILURE;
  }

  return result;
}
                                      

