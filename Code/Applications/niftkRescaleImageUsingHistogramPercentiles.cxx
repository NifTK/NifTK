/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

  =============================================================================*/

#include <itkLogHelper.h>
#include <niftkConversionUtils.h>

#include <itkCommandLineHelper.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkRescaleImageUsingHistogramPercentilesFilter.h>

#include <niftkRescaleImageUsingHistogramPercentilesCLP.h>

/*!
 * \file niftkRescaleImageUsingHistogramPercentiles.cxx
 * \page niftkRescaleImageUsingHistogramPercentiles
 * \section niftkRescaleImageUsingHistogramPercentilesSummary Rescale an image from input limits specified via percentiles of the input image histogram to some output range.
 *
 * This program uses ITK ImageFileReader to load an image, and then uses RescaleImageUsingHistogramPercentilesFilter to rescale its intensities before writing the output with ITK ImageFileWriter.
 *
 * \li Dimensions: 2,3.
 * \li Pixel type: Scalars only of unsigned char, char, unsigned short, short, unsigned int, int, unsigned long, long, float and double.
 *
 * \section niftkRescaleImageUsingHistogramPercentilesCaveats Caveats
 * \li None
 */
struct arguments
{
  bool flgVerbose;
  bool flgDebug;

  bool flgClipOutput;

  float inLowerPercentile;
  float inUpperPercentile;

  float outLowerLimit;
  float outUpperLimit;

  std::string inputImage;
  std::string outputImage;  

  arguments() {
    flgVerbose = false;
    flgDebug = false;

    flgClipOutput = false;

    inLowerPercentile = 0.;
    inUpperPercentile = 100.;

    outLowerLimit = 0.;
    outUpperLimit = 100.;
  }
};

template <int Dimension, class PixelType> 
int DoMain(arguments args)
{  
  typedef typename itk::Image< PixelType, Dimension >     InputImageType;   
  typedef typename itk::ImageFileReader< InputImageType > InputImageReaderType;
  typedef typename itk::ImageFileWriter< InputImageType > OutputImageWriterType;
  typedef typename itk::RescaleImageUsingHistogramPercentilesFilter<InputImageType, InputImageType> RescaleFilterType;

  typename InputImageReaderType::Pointer imageReader = InputImageReaderType::New();
  imageReader->SetFileName(args.inputImage);

  typename RescaleFilterType::Pointer filter = RescaleFilterType::New();
  filter->SetInput(imageReader->GetOutput());
  
  filter->SetInLowerPercentile( args.inLowerPercentile );
  filter->SetInUpperPercentile( args.inUpperPercentile );

  filter->SetOutLowerLimit( args.outLowerLimit );
  filter->SetOutUpperLimit( args.outUpperLimit );

  if ( args.flgVerbose )
  {
    filter->VerboseOn();
  }

  if ( args.flgDebug )
  {
    filter->DebugOn();
  }

  if ( args.flgClipOutput )
  {
    filter->ClipTheOutput();
  }

  typename OutputImageWriterType::Pointer imageWriter = OutputImageWriterType::New();
  imageWriter->SetFileName(args.outputImage);
  imageWriter->SetInput(filter->GetOutput());
  
  try
  {
    imageWriter->Update(); 
  }
  catch( itk::ExceptionObject & err ) 
  { 
    std::cerr << "ERROR: Failed: " << err << std::endl; 
    return EXIT_FAILURE;
  }                
  return EXIT_SUCCESS;
}

/**
 * \brief Takes the input image and recales it using RescaleImageUsingHistogramPercentilesFilter
 */
int main(int argc, char** argv)
{
  // To pass around command line args
  PARSE_ARGS;

  // To pass around command line args
  struct arguments args;

  args.inputImage=inputImage.c_str();
  args.outputImage=outputImage.c_str();

  args.inLowerPercentile = inLowerPercentile;
  args.inUpperPercentile = inUpperPercentile;

  args.outLowerLimit = outLowerLimit;
  args.outUpperLimit = outUpperLimit;

  args.flgVerbose = flgVerbose;
  args.flgDebug   = flgDebug;

  args.flgClipOutput = flgClipOutput;


  std::cout << "Input image:  " << args.inputImage << std::endl
            << "Output image: " << args.outputImage << std::endl
            << std::endl
            << "Rescaling intensities " << std::endl
            << "   from input range: " 
            << args.inLowerPercentile << "% to " << args.inUpperPercentile << "%"
            << std::endl
            << "   to output range:  "
            << args.outLowerLimit << " to " << args.outUpperLimit << std::endl
            << std::endl
            << "Verbose output?: "   << std::boolalpha << args.flgVerbose     
            << std::noboolalpha << std::endl
            << "Debugging output?: " << std::boolalpha << args.flgDebug       
            << std::noboolalpha << std::endl
            << "Clip the output image?: " << std::boolalpha << args.flgClipOutput  
            << std::noboolalpha << std::endl
            << std::endl;
            
  // Validate command line args
  if (args.inputImage.length() == 0 ||
      args.outputImage.length() == 0)
  {
    std::cerr << "ERROR: Input and output image file names must be specified." << std::endl;
    return EXIT_FAILURE;
  }

  int dims = itk::PeekAtImageDimensionFromSizeInVoxels(args.inputImage);
  if (dims != 2 && dims != 3)
  {
    std::cout << "ERROR: Unsupported image dimension" << std::endl;
    return EXIT_FAILURE;
  }
  else if (dims == 2)
  {
    std::cout << "Input is 2D" << std::endl;
  }
  else
  {
    std::cout << "Input is 3D" << std::endl;
  }
   
  int result;

  switch (itk::PeekAtComponentType(args.inputImage))
  {
  case itk::ImageIOBase::UCHAR:
    std::cout << "Input is UNSIGNED CHAR" << std::endl;
    if (dims == 2)
    {
      result = DoMain<2, unsigned char>(args);  
    }
    else
    {
      result = DoMain<3, unsigned char>(args);
    }
    break;
  case itk::ImageIOBase::CHAR:
    std::cout << "Input is CHAR" << std::endl;
    if (dims == 2)
    {
      result = DoMain<2, char>(args);  
    }
    else
    {
      result = DoMain<3, char>(args);
    }
    break;
  case itk::ImageIOBase::USHORT:
    std::cout << "Input is UNSIGNED SHORT" << std::endl;
    if (dims == 2)
    {
      result = DoMain<2, unsigned short>(args);  
    }
    else
    {
      result = DoMain<3, unsigned short>(args);
    }
    break;
  case itk::ImageIOBase::SHORT:
    std::cout << "Input is SHORT" << std::endl;
    if (dims == 2)
    {
      result = DoMain<2, short>(args);  
    }
    else
    {
      result = DoMain<3, short>(args);
    }
    break;
  case itk::ImageIOBase::UINT:
    std::cout << "Input is UNSIGNED INT" << std::endl;
    if (dims == 2)
    {
      result = DoMain<2, unsigned int>(args);  
    }
    else
    {
      result = DoMain<3, unsigned int>(args);
    }
    break;
  case itk::ImageIOBase::INT:
    std::cout << "Input is INT" << std::endl;
    if (dims == 2)
    {
      result = DoMain<2, int>(args);  
    }
    else
    {
      result = DoMain<3, int>(args);
    }
    break;
  case itk::ImageIOBase::ULONG:
    std::cout << "Input is UNSIGNED LONG" << std::endl;
    if (dims == 2)
    {
      result = DoMain<2, unsigned long>(args);  
    }
    else
    {
      result = DoMain<3, unsigned long>(args);
    }
    break;
  case itk::ImageIOBase::LONG:
    std::cout << "Input is LONG" << std::endl;
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
    std::cout << "Input is FLOAT" << std::endl;
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
    std::cout << "Input is DOUBLE" << std::endl;
    if (dims == 2)
    {
      result = DoMain<2, double>(args);  
    }
    else
    {
      result = DoMain<3, double>(args);
    }
    break;
  default:
    std::cerr << "ERROR: non standard pixel format" << std::endl;
    return EXIT_FAILURE;
  }
  return result;
}
