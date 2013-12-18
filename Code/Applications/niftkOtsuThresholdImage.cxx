/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <math.h>
#include <float.h>

#include <niftkConversionUtils.h>
#include <niftkCommandLineParser.h>
#include <itkCommandLineHelper.h>

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImage.h>

#include <itkOtsuThresholdImageFilter.h>
#include <itkInvertIntensityBetweenMaxAndMinImageFilter.h>
#include <itkImageRegionConstIteratorWithIndex.h>
#include <itkImageRegionConstIterator.h>

#include <niftkOtsuThresholdImageCLP.h>


// -------------------------------------------------------------------------
// arguments
// -------------------------------------------------------------------------

struct arguments
{
  bool flgInvertOutputMask;

  std::string fileInputImage;
  std::string fileInputMask;
  std::string fileOutputMask;

  arguments() {
    flgInvertOutputMask = false;
  }
};


// -------------------------------------------------------------------------
// DoMain(arguments args)
// -------------------------------------------------------------------------

template <int Dimension, class OutputPixelType>
int DoMain(arguments &args)
{

  // Read the input image
  // ~~~~~~~~~~~~~~~~~~~~

  typedef float InputPixelType;
  typedef itk::Image<InputPixelType, Dimension> InputImageType;

  typedef itk::ImageFileReader< InputImageType > FileReaderType;

  typename FileReaderType::Pointer imageReader = FileReaderType::New();

  imageReader->SetFileName( args.fileInputImage );
  

  try
  { 
    std::cout << "Reading the input image" << std::endl;
    imageReader->UpdateLargestPossibleRegion();
  }
  catch (itk::ExceptionObject &ex)
  { 
    std::cout << ex << std::endl;
    return EXIT_FAILURE;
  }

  typename InputImageType::Pointer image = imageReader->GetOutput();
  image->DisconnectPipeline();  

  typename InputImageType::RegionType  inputRegion  = image->GetLargestPossibleRegion(); 
  typename InputImageType::PointType   inputOrigin  = image->GetOrigin(); 
  typename InputImageType::SpacingType inputSpacing = image->GetSpacing(); 

  std::cout << "Origin: "  << inputOrigin << std::endl
            << "Spacing: " << inputSpacing << std::endl
            << inputRegion << std::endl;


  // Read the input mask?
  // ~~~~~~~~~~~~~~~~~~~~

  typedef unsigned char MaskPixelType;
  typedef itk::Image<MaskPixelType, Dimension> MaskImageType;

  typename MaskImageType::Pointer inMask = 0;

  if ( args.fileInputMask.length() != 0 ) 
  {
    
    typedef itk::ImageFileReader< MaskImageType > MaskReaderType;

    typename MaskReaderType::Pointer maskReader = MaskReaderType::New();

    maskReader->SetFileName( args.fileInputMask );

    try
    { 
      std::cout << "Reading the mask image" << std::endl;
      maskReader->UpdateLargestPossibleRegion();
    }
    catch (itk::ExceptionObject &ex)
    { 
      std::cout << ex << std::endl;
      return EXIT_FAILURE;
    }

    inMask = maskReader->GetOutput();

    typename MaskImageType::RegionType  maskRegion  = inMask->GetLargestPossibleRegion(); 
    typename MaskImageType::PointType   maskOrigin  = inMask->GetOrigin(); 
    typename MaskImageType::SpacingType maskSpacing = inMask->GetSpacing(); 
    
    std::cout << "Origin: "  << maskOrigin << std::endl
              << "Spacing: " << maskSpacing << std::endl
              << maskRegion  << std::endl;
    
    if ( inputRegion == maskRegion )
      std::cout << "Input image and mask regions coincide" << std::endl;
    else
    {
      std::cout << "Input image and mask regions do not coincide" << std::endl;
      return EXIT_FAILURE;
    }
  }


  // Calculate the indices of the max and min intensities in the input image
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef typename InputImageType::IndexType IndexType;

  InputPixelType value;

  IndexType minIndex;
  IndexType maxIndex;

  double min = std::numeric_limits<double>::max();
  double max = std::numeric_limits<double>::min();

  itk::ImageRegionConstIteratorWithIndex< InputImageType > itImage( image, 
                                                                    image->GetLargestPossibleRegion() );

  if ( inMask )
  {

    itk::ImageRegionConstIterator< MaskImageType > itInputMask( inMask, 
                                                           inMask->GetLargestPossibleRegion() );

    for ( itImage.GoToBegin(), itInputMask.GoToBegin(); 
          ! itImage.IsAtEnd(); 
          ++itImage, ++itInputMask )
    {
      if ( itInputMask.Get() )
      {
        
        value = itImage.Get();
        
        if (value > max) 
        {
          max = value;
          maxIndex = itImage.GetIndex();
        }
        else if (value < min)
        {
          min = value;
          minIndex = itImage.GetIndex();
        }
      }
    }
  }

  else
  {
    for ( itImage.GoToBegin(); ! itImage.IsAtEnd(); ++itImage )
    {
      value = itImage.Get();
      
      if (value > max) 
      {
        max = value;
        maxIndex = itImage.GetIndex();
      }
      else if (value < min)
      {
        min = value;
        minIndex = itImage.GetIndex();
      }
    }
  }

  std::cout << "Minimum image intensity: " << min
            << " at: " << minIndex << std::endl
            << "Maximum image intensity: " << max
            << " at: " << maxIndex << std::endl;


  // Create a mask with the itk::OtsuThresholdImageFilterType
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typename MaskImageType::Pointer outMask;

  typedef itk::OtsuThresholdImageFilter< InputImageType, 
                                         MaskImageType, 
                                         MaskImageType > OtsuThresholdImageFilterType;

  typename OtsuThresholdImageFilterType::Pointer thresholder = OtsuThresholdImageFilterType::New();
    
  thresholder->SetInput( image );

  if ( inMask ) 
  {
    thresholder->SetMaskImage( inMask );
  }

  // Calculate the threshold
    
  try
  {
    std::cout << "Thresholding to obtain image mask" << std::endl;
    thresholder->Update();
  }
  catch (itk::ExceptionObject &e)
  {
    std::cerr << e << std::endl;
  }

  outMask = thresholder->GetOutput();
  outMask->DisconnectPipeline();  
    

  // If the image max intensity is not in the mask then invert it
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  bool flgInvertMask = false;

  if ( ( ! outMask->GetPixel( maxIndex ) ) && ( outMask->GetPixel( minIndex ) ) )
  {
    std::cout << "WARNING: Maximum image intensity is not inside the calculated mask" << std::endl
              << "   and minimum intensity is not outside so the mask will be inverted." << std::endl;
    if ( ! args.flgInvertOutputMask )
      flgInvertMask = true;
  }
  else if ( args.flgInvertOutputMask )
  {
      flgInvertMask = true;
  }


  if ( flgInvertMask ) 
  {
    itk::ImageRegionIterator< MaskImageType > itOutputMask( outMask, 
                                                            outMask->GetLargestPossibleRegion() );

    if ( inMask )
    {
      
      itk::ImageRegionConstIterator< MaskImageType > itInputMask( inMask, 
                                                                  inMask->GetLargestPossibleRegion() );
      
      for ( itOutputMask.GoToBegin(), itInputMask.GoToBegin(); 
            ! itOutputMask.IsAtEnd(); 
            ++itOutputMask, ++itInputMask )
      {
        if ( itInputMask.Get() )
        {        
          itOutputMask.Set( itOutputMask.Get() ? 0 : 255 );
        } 
      }
    }

    else
    {
      for ( itOutputMask.GoToBegin(); ! itOutputMask.IsAtEnd(); ++itOutputMask )
      {
        itOutputMask.Set( itOutputMask.Get() ? 0 : 255 );
      }
    }
  }


  // Write the mask image to a file?
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( args.fileOutputMask.length() != 0 ) 
  {
    typedef itk::Image<MaskPixelType, Dimension> OutputMaskType;

    typedef itk::ImageFileWriter< OutputMaskType > FileWriterType;

    typename FileWriterType::Pointer writer = FileWriterType::New();

    writer->SetFileName( args.fileOutputMask );
    writer->SetInput( outMask );

    try
    {
      std::cout << "Writing the mask to image: " << args.fileOutputMask << std::endl;
      writer->Update();
    }
    catch (itk::ExceptionObject &e)
    {
      std::cerr << e << std::endl;
      return EXIT_FAILURE;
    }
  }
  
  return EXIT_SUCCESS;
}


// -------------------------------------------------------------------------
// main( int argc, char *argv[] )
// -------------------------------------------------------------------------

int main( int argc, char *argv[] )
{
  // To pass around command line args
  arguments args;

  // Validate command line args
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~

  PARSE_ARGS;

  if ( fileInputImage.length() == 0 || fileOutputMask.length() == 0 )
  {
    commandLine.getOutput()->usage(commandLine);
    return EXIT_FAILURE;
  }

  args.fileInputImage = fileInputImage;
  args.fileInputMask  = fileInputMask;
  args.fileOutputMask = fileOutputMask;

  args.flgInvertOutputMask = flgInvertOutputMask;

  int dims = itk::PeekAtImageDimensionFromSizeInVoxels(args.fileInputImage);

  if ( (dims != 2) && (dims != 3) )
  {
    std::cout << "ERROR: Unsupported image dimension, image must be 2D or 3D" << std::endl;
    return EXIT_FAILURE;
  }
  
  int result;

  if ( dims == 2 )
  {
    switch (itk::PeekAtComponentType(args.fileInputImage))
    {
    case itk::ImageIOBase::UCHAR:
      result = DoMain<2, unsigned char>(args);
      break;

    case itk::ImageIOBase::CHAR:
      result = DoMain<2, char>(args);
      break;

    case itk::ImageIOBase::USHORT:
      result = DoMain<2, unsigned short>(args);
      break;

    case itk::ImageIOBase::SHORT:
      result = DoMain<2, short>(args);
      break;

    case itk::ImageIOBase::UINT:
      result = DoMain<2, unsigned int>(args);
      break;

    case itk::ImageIOBase::INT:
      result = DoMain<2, int>(args);
      break;

    case itk::ImageIOBase::ULONG:
      result = DoMain<2, unsigned long>(args);
      break;

    case itk::ImageIOBase::LONG:
      result = DoMain<2, long>(args);
      break;

    case itk::ImageIOBase::FLOAT:
      result = DoMain<2, float>(args);
      break;

    case itk::ImageIOBase::DOUBLE:
      result = DoMain<2, double>(args);
      break;

    default:
      std::cerr << "ERROR: Unsupported pixel format" << std::endl;
      return EXIT_FAILURE;
    }
  }
  else 
  {

    switch (itk::PeekAtComponentType(args.fileInputImage))
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
      std::cerr << "ERROR: Unsupported pixel format" << std::endl;
      return EXIT_FAILURE;
    }
  }

  return result;
}
