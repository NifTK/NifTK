/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <niftkLogHelper.h>
#include <niftkConversionUtils.h>
#include <itkCommandLineHelper.h>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkNifTKImageIOFactory.h>
#include <itkAbsImageFilter.h>

/*!
 * \file niftkAbsImageFilter.cxx
 * \page niftkAbsImageFilter
 * \section niftkAbsImageFilterSummary Runs the ITK AbsImageFilter on a single image to output the absolute value image.
 *
 * This program uses ITK ImageFileReaders to load an image, then ITK AbsImageFilter to take the absolute value of each voxel
 * and then writes the output using ITK ImageFileWriter.
 *
 * \li Dimensions: 2,3.
 * \li Pixel type: Scalars only of char, short, int, long, float and double.
 *
 * \section niftkAbsImageFilterCaveat Caveats
 * \li None.
 */
void Usage(char *exec)
  {
    niftk::LogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl;
    std::cout << "  Runs the ITK AbsImageFilter on a single 2D or 3D image to output the absolute value image, useful for displaying the image properly in MIDAS" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "  " << exec << " -i inputFileName -o outputFileName" << std::endl;
    std::cout << "  " << std::endl;
    return;
  }

struct arguments
{
  std::string inputImage;
  std::string outputImage;  
};

template <int Dimension, class PixelType> 
int DoMain(arguments args)
{  
  typedef typename itk::Image< PixelType, Dimension >     InputImageType;   
  typedef typename itk::ImageFileReader< InputImageType > InputImageReaderType;
  typedef typename itk::ImageFileWriter< InputImageType > OutputImageWriterType;
  
  typename InputImageReaderType::Pointer  imageReader = InputImageReaderType::New();
  typename OutputImageWriterType::Pointer imageWriter = OutputImageWriterType::New();
  
  imageReader->SetFileName( args.inputImage );
  imageWriter->SetFileName( args.outputImage );
  
  typedef typename itk::AbsImageFilter< InputImageType, InputImageType > AbsImageFilterType; 
  typename AbsImageFilterType::Pointer absImageFilter  = AbsImageFilterType::New();
  
  absImageFilter->SetInput(imageReader->GetOutput()); 
  imageWriter->SetInput(absImageFilter->GetOutput()); 
  
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
 * \brief Take the abs value of an image, for displaying the image properly in Midas. 
 */
int main(int argc, char** argv)
{
  itk::NifTKImageIOFactory::Initialize();

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
      std::cout << "Unsupported image dimension" << std::endl;
      return EXIT_FAILURE;
    }
  
  int result;

  switch (itk::PeekAtComponentType(args.inputImage))
    {
    case itk::ImageIOBase::CHAR:
      if (dims == 2)
        {
          result = DoMain<2, char>(args);  
        }
      else
        {
          result = DoMain<3, char>(args);
        }
      break;
    case itk::ImageIOBase::SHORT:
      if (dims == 2)
        {
          result = DoMain<2, short>(args);  
        }
      else
        {
          result = DoMain<3, short>(args);
        }
      break;
    case itk::ImageIOBase::INT:
      if (dims == 2)
        {
          result = DoMain<2, int>(args);  
        }
      else
        {
          result = DoMain<3, int>(args);
        }
      break;
    case itk::ImageIOBase::LONG:
      if (dims == 2)
        {
          result = DoMain<2, long>(args);  
        }
      else
        {
          result = DoMain<3, long>(args);
        }
      break;
    case itk::ImageIOBase::FLOAT:
      if (dims == 2)
        {
          result = DoMain<2, float>(args);  
        }
      else
        {
          result = DoMain<3, float>(args);
        }
      break;
    case itk::ImageIOBase::DOUBLE:
      if (dims == 2)
        {
          result = DoMain<2, double>(args);  
        }
      else
        {
          result = DoMain<3, double>(args);
        }
      break;
    case itk::ImageIOBase::UCHAR:
    case itk::ImageIOBase::USHORT:
    case itk::ImageIOBase::UINT:
    case itk::ImageIOBase::ULONG:
      std::cerr << "You shouldn't run this on unsigned data types, so Im defaulting to float"  << std::endl;
      if (dims == 2)
        {
          result = DoMain<2, float>(args);  
        }
      else
        {
          result = DoMain<3, float>(args);
        }      
      break;
    default:
      std::cerr << "non standard pixel format" << std::endl;
      return EXIT_FAILURE;
    }
  return result;
}
