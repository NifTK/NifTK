/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

  =============================================================================*/

#include <itkLogHelper.h>
#include <ConversionUtils.h>
#include <itkCommandLineHelper.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkInvertIntensityBetweenMaxAndMinImageFilter.h>
#include <itkLogNonZeroIntensitiesImageFilter.h>
#include <itkMinimumMaximumImageCalculator.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkCastImageFilter.h>

#include <niftkLogInvertImageCLP.h>

/*!
 * \file niftkLogInvertImage.cxx
 * \page niftkLogInvertImage
 * \section niftkLogInvertImageSummary Runs filters InvertIntensityBetweenMaxAndMinImageFilter and LogNonZeroIntensitiesImageFilter on an image, computing Inew = Log( Imax - I + Imin )
 *
 * This program uses ITK ImageFileReader to load an image, and then uses InvertIntensityBetweenMaxAndMinImageFilter to invert its intensities and LogNonZeroIntensitiesImageFilter to compute the logarithm before writing the output with ITK ImageFileWriter.
 *
 * \li Dimensions: 2,3.
 * \li Pixel type: Scalars only of unsigned char, char, unsigned short, short, unsigned int, int, unsigned long, long, float and double.
 *
 * \section niftkLogInvertImageCaveats Caveats
 * \li None
 */
struct arguments
{
  bool flgPreserveImageRange;

  std::string inputImage;
  std::string outputImage;  
};

template <int Dimension, class PixelType> 
int DoMain(arguments args)
{  
  typedef float InternalPixelType;
  typedef typename itk::Image< PixelType, Dimension >         InputImageType; 
  typedef typename itk::Image< InternalPixelType, Dimension > InternalImageType; 
  
  typedef typename itk::ImageFileReader< InternalImageType > InputImageReaderType;
  typedef typename itk::ImageFileWriter< InputImageType > OutputImageWriterType;
  typedef typename itk::InvertIntensityBetweenMaxAndMinImageFilter<InternalImageType> InvertFilterType;
  typedef typename itk::LogNonZeroIntensitiesImageFilter<InternalImageType, InternalImageType> LogFilterType;
  typedef typename itk::MinimumMaximumImageCalculator<InternalImageType> MinimumMaximumImageCalculatorType;
  typedef typename itk::RescaleIntensityImageFilter< InternalImageType, InternalImageType > RescalerType;
  typedef typename itk::CastImageFilter< InternalImageType, InputImageType > CastingFilterType;

  // Read the image

  typename InputImageReaderType::Pointer imageReader = InputImageReaderType::New();
  imageReader->SetFileName(args.inputImage);
  imageReader->Update();


  // Compute the log

  typename LogFilterType::Pointer logfilter = LogFilterType::New();
  logfilter->SetInput(imageReader->GetOutput());
  logfilter->UpdateLargestPossibleRegion();


  // Invert the image

  typename InvertFilterType::Pointer invfilter = InvertFilterType::New();
  invfilter->SetInput(logfilter->GetOutput());
  invfilter->UpdateLargestPossibleRegion();


  // Instantiate the filter to cast the image to the original image type

  typename CastingFilterType::Pointer caster = CastingFilterType::New();


  // Restore the image range to that of the original?

  if ( args.flgPreserveImageRange )
  {
    typename MinimumMaximumImageCalculatorType::Pointer 
      imageRangeCalculator = MinimumMaximumImageCalculatorType::New();

    imageRangeCalculator->SetImage( imageReader->GetOutput() );
    imageRangeCalculator->Compute();

    typename RescalerType::Pointer intensityRescaler = RescalerType::New();
    intensityRescaler->SetInput(invfilter->GetOutput());  

    intensityRescaler->SetOutputMinimum( 
      static_cast< PixelType >( imageRangeCalculator->GetMinimum() ) );
    intensityRescaler->SetOutputMaximum( 
      static_cast< PixelType >( imageRangeCalculator->GetMaximum() ) );

    intensityRescaler->UpdateLargestPossibleRegion();

    caster->SetInput( intensityRescaler->GetOutput() );
  }
  else
  {
    caster->SetInput( invfilter->GetOutput() );
  }

  caster->UpdateLargestPossibleRegion();


  // Instantiate the writer

  typename OutputImageWriterType::Pointer imageWriter = OutputImageWriterType::New();
  imageWriter->SetFileName(args.outputImage);
  imageWriter->SetInput( caster->GetOutput() );


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
 * \brief Takes the input image and inverts it using InvertIntensityBetweenMaxAndMinImageFilter
 */
int main(int argc, char** argv)
{
  // To pass around command line args
  PARSE_ARGS;

  // To pass around command line args
  struct arguments args;
  args.inputImage=inputImage.c_str();
  args.outputImage=outputImage.c_str();
  args.flgPreserveImageRange = flgPreserveImageRange;

  std::cout << "Input image:  " << args.inputImage << std::endl
            << "Output image: " << args.outputImage << std::endl;

  // Validate command line args
  if (args.inputImage.length() == 0 ||
      args.outputImage.length() == 0)
  {
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
