/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "itkLogHelper.h"
#include "ConversionUtils.h"
#include "itkCommandLineHelper.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkRegionOfInterestImageFilter.h"

#include "itkImage.h"


void Usage(char *exec)
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl;
    std::cout << "  Runs ITK ExtractRegion image function" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "  " << exec << " -i inputFileName -o outputFileName [options]" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "*** [mandatory] ***" << std::endl << std::endl;
    std::cout << "    -i    <filename>        Input image" << std::endl;
    std::cout << "    -o    <filename>        Output image" << std::endl << std::endl;
    std::cout << "    -sx    <int>            Size of region in x-axis (vox)" << std::endl << std::endl;
    std::cout << "    -sy    <int>            Size of region in y-axis (vox)" << std::endl << std::endl;
    std::cout << "    -sz    <int>            Size of region in z-axis (vox)" << std::endl << std::endl;
    std::cout << "    -ix    <int>            Start index of region in x-axis (vox)" << std::endl << std::endl;
    std::cout << "    -iy    <int>            Start index of region in y-axis (vox)" << std::endl << std::endl;
    std::cout << "    -iz    <int>            Start index of region in z-axis (vox)" << std::endl << std::endl;
    std::cout << "*** [options]   ***" << std::endl << std::endl;   
  }

struct arguments
{
  arguments()
  :inputImage("")
  ,outputImage("")
  ,sx(1)
  ,sy(1)
  ,sz(1)
  ,ix(0)
  ,iy(0)
  ,iz(0)
  {}

  std::string inputImage;
  std::string outputImage;
  int sx;
  int sy;
  int sz;
  int ix;
  int iy;
  int iz;
};

template <int Dimension, class PixelType> 
int ExtractRegion(arguments args)
{  
    typedef   typename itk::Image<PixelType, Dimension >     ImageType;
    typedef   typename itk::ImageFileReader< ImageType > InputImageReaderType;
    typedef   typename itk::ImageFileWriter< ImageType > OutputImageWriterType;
    typedef   typename itk::RegionOfInterestImageFilter<ImageType, ImageType> ROIImageFilter;
    
    typename InputImageReaderType::Pointer imageReader = InputImageReaderType::New();
    imageReader->SetFileName(args.inputImage);

    typename ROIImageFilter::Pointer filter = ROIImageFilter::New();
    typename ROIImageFilter::RegionType region;
    typename ROIImageFilter::SizeType size;
    typename ROIImageFilter::IndexType start;
    
    start[0] = args.ix;
    start[1] = args.iy;
    if (Dimension == 3)
    {
      start[2] = args.iz;
    }
  
    size[0] = args.sx;
    size[1] = args.sy;
    if (Dimension == 3)
    {
      size[2] = args.sz;
    }

    region.SetSize(  size  );
    region.SetIndex( start );

    filter->SetInput(imageReader->GetOutput());
    filter->SetRegionOfInterest(region);
    filter->Update();

    typename OutputImageWriterType::Pointer imageWriter = OutputImageWriterType::New();
    imageWriter->SetFileName(args.outputImage);
    imageWriter->SetInput(filter->GetOutput());
  
    try
    {
      imageWriter->Update();
    }
    catch( itk::ExceptionObject & err )
    {
      std::cerr << "Failed: " << err << std::endl;
      return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

/**
 * \brief Takes image1 and image2 and adds them together
 */
int main(int argc, char** argv)
{
  // To pass around command line args
  struct arguments args;
  
  

  // Parse command line args
  for(int i=1; i < argc; i++){
    if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 || strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--h")==0){
      Usage(argv[0]);
      return -1;
    }
    else if(strcmp(argv[i], "-i") == 0){
      args.inputImage=argv[++i];
      std::cout << "Set -i=" << args.inputImage << std::endl;
    }
    
    else if(strcmp(argv[i], "-o") == 0){
      args.outputImage=argv[++i];
      std::cout << "Set -o=" << args.outputImage << std::endl;
    }
   
    else if(strcmp(argv[i], "-sx") == 0){
      args.sx=atoi(argv[++i]);
      std::cout << "Set -sx=" << niftk::ConvertToString(args.sx) << std::endl;
    }
    else if(strcmp(argv[i], "-sy") == 0){
      args.sy=atoi(argv[++i]);
      std::cout << "Set -sy=" << niftk::ConvertToString(args.sy) << std::endl;
    }
    else if(strcmp(argv[i], "-sz") == 0){
      args.sz=atoi(argv[++i]);
      std::cout << "Set -sz=" << niftk::ConvertToString(args.sz) << std::endl;
    }
    else if(strcmp(argv[i], "-ix") == 0){
      args.ix=atoi(argv[++i]);
      std::cout << "Set -ix=" << niftk::ConvertToString(args.ix) << std::endl;
    }
    else if(strcmp(argv[i], "-iy") == 0){
      args.iy=atoi(argv[++i]);
      std::cout << "Set -iy=" << niftk::ConvertToString(args.iy) << std::endl;
    }
    else if(strcmp(argv[i], "-iz") == 0){
      args.iz=atoi(argv[++i]);
      std::cout << "Set -iz=" << niftk::ConvertToString(args.iz) << std::endl;
    }
    else {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return -1;
    }            
  }

  // Validate command line args
  if (args.inputImage.length() == 0 || args.outputImage.length() == 0)
    {
      Usage(argv[0]);
      return EXIT_FAILURE;
    }

  int dims = itk::PeekAtImageDimension(args.inputImage);
  if (dims != 2 && dims != 3)
    {
      std::cout << "Unsuported image dimension" << std::endl;
      return EXIT_FAILURE;
    }
  
  int result;

  switch (itk::PeekAtComponentType(args.inputImage))
    {
    case itk::ImageIOBase::UCHAR:
      if (dims == 2)
        {
          result = ExtractRegion<2, unsigned char>(args);  
        }
      else
        {
          result = ExtractRegion<3, unsigned char>(args);
        }
      break;
    case itk::ImageIOBase::CHAR:
      if (dims == 2)
        {
          result = ExtractRegion<2, char>(args);  
        }
      else
        {
          result = ExtractRegion<3, char>(args);
        }
      break;
    case itk::ImageIOBase::USHORT:
      if (dims == 2)
        {
          result = ExtractRegion<2, unsigned short>(args);  
        }
      else
        {
          result = ExtractRegion<3, unsigned short>(args);
        }
      break;
    case itk::ImageIOBase::SHORT:
      if (dims == 2)
        {
          result = ExtractRegion<2, short>(args);  
        }
      else
        {
          result = ExtractRegion<3, short>(args);
        }
      break;
    case itk::ImageIOBase::UINT:
      if (dims == 2)
        {
          result = ExtractRegion<2, unsigned int>(args);  
        }
      else
        {
          result = ExtractRegion<3, unsigned int>(args);
        }
      break;
    case itk::ImageIOBase::INT:
      if (dims == 2)
        {
          result = ExtractRegion<2, int>(args);  
        }
      else
        {
          result = ExtractRegion<3, int>(args);
        }
      break;
    case itk::ImageIOBase::ULONG:
      if (dims == 2)
        {
          result = ExtractRegion<2, unsigned long>(args);  
        }
      else
        {
          result = ExtractRegion<3, unsigned long>(args);
        }
      break;
    case itk::ImageIOBase::LONG:
      if (dims == 2)
        {
          result = ExtractRegion<2, long>(args);  
        }
      else
        {
          result = ExtractRegion<3, long>(args);
        }
      break;
    case itk::ImageIOBase::FLOAT:
      if (dims == 2)
        {
          result = ExtractRegion<2, float>(args);  
        }
      else
        {
          result = ExtractRegion<3, float>(args);
        }
      break;
    case itk::ImageIOBase::DOUBLE:
      if (dims == 2)
        {
          result = ExtractRegion<2, double>(args);  
        }
      else
        {
          result = ExtractRegion<3, double>(args);
        }
      break;
    default:
      std::cerr << "non standard pixel format" << std::endl;
      return EXIT_FAILURE;
    }
  return result;
}
