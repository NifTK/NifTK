/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-24 17:44:42 +0000 (Thu, 24 Nov 2011) $
 Revision          : $Revision: 7864 $
 Last modified by  : $Author: kkl $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#include "itkLogHelper.h"
#include "ConversionUtils.h"
#include "itkCommandLineHelper.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkExtractImageFilter.h"

/*!
 * \file niftkExtract2DSliceFrom3DImage.cxx
 * \page niftkExtract2DSliceFrom3DImage
 * \section niftkExtract2DSliceFrom3DImageSummary Runs the ITK ExtractImageFilter.
 */
void Usage(char *exec)
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl;
    std::cout << "  Runs the ITK ExtractImageFilter." << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "  " << exec << " -i inputFileName -o outputFileName [options]" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "*** [mandatory] ***" << std::endl << std::endl;
    std::cout << "    -i    <filename>        Input image " << std::endl;
    std::cout << "    -o    <filename>        Output image" << std::endl << std::endl;      
    std::cout << "*** [options]   ***" << std::endl << std::endl;   
    std::cout << "    -size [x y z]           Extracted size. All 3 indexes must be specified. One index must be zero." << std::endl;
    std::cout << "    -index [x y z]          Index of starting position. All 3 indexes must be specified. " << std::endl;
  }

struct arguments
{
  std::string inputImage;
  std::string outputImage;
  int size[3];
  int index[3];  
};

template <class PixelType> 
int DoMain(arguments args)
{
  const   unsigned int InputDimension = 3;
  const   unsigned int OutputDimension = 2;

  typedef itk::Image< PixelType, InputDimension >                  InputImageType;
  typedef itk::Image< PixelType, OutputDimension >                 OutputImageType; 
  typedef itk::ImageFileReader< InputImageType >                   InputImageReaderType;
  typedef itk::ImageFileWriter< OutputImageType >                  OutputImageWriterType;
  typedef itk::ExtractImageFilter<InputImageType, OutputImageType> ExtractImageFilterType;
  
  typename InputImageReaderType::Pointer imageReader = InputImageReaderType::New();
  imageReader->SetFileName(args.inputImage);
  
  typename InputImageType::SizeType sizeType;
  typename InputImageType::IndexType indexType;
  typename InputImageType::RegionType regionType;
  
  for (unsigned int i = 0; i < InputDimension; i++)
    {
      sizeType[i] = args.size[i];
      indexType[i] = args.index[i];
    }
  regionType.SetSize(sizeType);
  regionType.SetIndex(indexType);
  
  typename ExtractImageFilterType::Pointer filter = ExtractImageFilterType::New();
  filter->SetInput(imageReader->GetOutput());
  filter->SetExtractionRegion(regionType);
  
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
 * \brief Takes image and does shifting/scaling in ITK style.
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
    else if(strcmp(argv[i], "-size") == 0){
      args.size[0]=atoi(argv[++i]);
      args.size[1]=atoi(argv[++i]);
      args.size[2]=atoi(argv[++i]);
      std::cout << "Set -size=" \
          << niftk::ConvertToString(args.size[0]) \
          << ", " << niftk::ConvertToString(args.size[1]) \
          << ", " << niftk::ConvertToString(args.size[2]) << std::endl;
    }
    else if(strcmp(argv[i], "-index") == 0){
      args.index[0]=atoi(argv[++i]);
      args.index[1]=atoi(argv[++i]);
      args.index[2]=atoi(argv[++i]);
      std::cout << "Set -index=" \
          << niftk::ConvertToString(args.index[0]) \
          << ", " << niftk::ConvertToString(args.index[1]) \
          << ", " << niftk::ConvertToString(args.index[2]) << std::endl;
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

  switch (itk::PeekAtComponentType(args.inputImage))
    {
    case itk::ImageIOBase::UCHAR:
      DoMain<unsigned char>(args);
      break;
    case itk::ImageIOBase::CHAR:
      DoMain<char>(args);
      break;
    case itk::ImageIOBase::USHORT:
      DoMain<unsigned short>(args);
      break;
    case itk::ImageIOBase::SHORT:
      DoMain<short>(args);
      break;
    case itk::ImageIOBase::UINT:
      DoMain<unsigned int>(args);
      break;
    case itk::ImageIOBase::INT:
      DoMain<int>(args);
      break;
    case itk::ImageIOBase::ULONG:
      DoMain<unsigned long>(args);
      break;
    case itk::ImageIOBase::LONG:
      DoMain<long>(args);
      break;
    case itk::ImageIOBase::FLOAT:
      DoMain<float>(args);
      break;
    case itk::ImageIOBase::DOUBLE:
      DoMain<double>(args);
      break;
    default:
      std::cerr << "non standard pixel format" << std::endl;
      return EXIT_FAILURE;
    }

  return EXIT_SUCCESS; 
}
