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
#include <itkHistogramMatchingImageFilter.h>

#include <niftkHistogramMatchingImageFilterCLP.h>

/*!
 * \file niftkHistogramMatchingImageFilter.cxx
 * \page niftkHistogramMatchingImageFilter
 * \section niftkHistogramMatchingImageFilterSummary Normalize the
 * grayscale values between two images by histogram matching.
 *
 * Applies ITK filter HistogramMatchingImageFilter to an image to
 * normalize the grey-scale values of a source image based on the
 * grey-scale values of a reference image. This filter uses a
 * histogram matching technique where the histograms of the two images
 * are matched only at a specified number of quantile values.
 *
 * \li Dimensions: 2,3.
 * \li Pixel type: Scalars only of unsigned char, char, unsigned short, short, unsigned int, int, unsigned long, long, float and double.
 *
 * \section niftkHistogramMatchingImageFilterCaveats Caveats
 * \li File sizes not checked.
 */
struct arguments
{
  std::string fileSourceImage;
  std::string fileReferenceImage;
  std::string fileOutputImage;  
};

template <int Dimension, class PixelType> 
int DoMain(arguments args)
{  
  typedef typename itk::Image< PixelType, Dimension >     InputImageType;   
  typedef typename itk::ImageFileReader< InputImageType > InputImageReaderType;
  typedef typename itk::ImageFileWriter< InputImageType > OutputImageWriterType;
  typedef typename itk::HistogramMatchingImageFilter<InputImageType, InputImageType> HistogramMatchFilterType;

  typename InputImageReaderType::Pointer imageReaderSource = InputImageReaderType::New();
  imageReaderSource->SetFileName(args.fileSourceImage);

  typename InputImageReaderType::Pointer imageReaderReference = InputImageReaderType::New();
  imageReaderReference->SetFileName(args.fileReferenceImage);

  typename HistogramMatchFilterType::Pointer filter = HistogramMatchFilterType::New();
  filter->SetSourceImage(imageReaderSource->GetOutput());
  filter->SetReferenceImage(imageReaderReference->GetOutput());

  filter->ThresholdAtMeanIntensityOn();

  typename OutputImageWriterType::Pointer imageWriter = OutputImageWriterType::New();
  imageWriter->SetFileName(args.fileOutputImage);
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
 * \brief Normalize the grayscale values between two images by histogram matching.
 */
int main(int argc, char** argv)
{
  // To pass around command line args
  PARSE_ARGS;

  // To pass around command line args
  struct arguments args;
  args.fileSourceImage    = fileSourceImage;
  args.fileReferenceImage = fileReferenceImage;
  args.fileOutputImage    = fileOutputImage;

  // Validate command line args
  if (args.fileSourceImage.length()    == 0 ||
      args.fileReferenceImage.length() == 0 ||
      args.fileOutputImage.length()    == 0)
  {
    return EXIT_FAILURE;
  }

  int dims = itk::PeekAtImageDimensionFromSizeInVoxels( args.fileSourceImage );

  if (dims != 2 && dims != 3)
  {
    std::cout << "ERROR: Unsupported image dimension" << std::endl;
    return EXIT_FAILURE;
  }
  
  int result;

  switch (itk::PeekAtComponentType(args.fileSourceImage))
  {
  case itk::ImageIOBase::UCHAR:
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
  default:
    std::cerr << "non standard pixel format" << std::endl;
    return EXIT_FAILURE;
  }

  return result;
}
