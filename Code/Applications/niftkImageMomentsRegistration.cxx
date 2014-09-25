/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

  =============================================================================*/

#include <niftkConversionUtils.h>
#include <niftkImageMomentsRegistrationCLP.h>

#include <itkLogHelper.h>
#include <itkCommandLineHelper.h>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkTransformFileWriter.h>
#include <itkImageMomentsCalculator.h>
#include <itkCropTargetImageWhereSourceImageNonZero.h>
#include <itkResampleImageFilter.h>
#include <itkScaleTransform.h>

/*!
 * \file niftkImageMomentsRegistration.cxx
 * \page niftkImageMomentsRegistration
 * \section niftkImageMomentsRegistrationSummary Registers two images using the ITK ImageMomentsCalculator by calculating a rigid plus scale transformation.
 *
 * \li Dimensions: 2,3.
 * \li Pixel type: Scalar images only that are converted to float on input.
 *
 * \section niftkImageMomentsRegistrationCaveats Caveats
 * \li None
 */

struct arguments
{
  std::string fileFixedImage;
  std::string fileMovingImage;

  std::string fileFixedMask;
  std::string fileMovingMask;     

  std::string fileOutputImage;

  std::string fileOutputTransformFile; 
};

template <int Dimension>
int DoMain(arguments args)
{
  typedef  float           PixelType;
  typedef  double          ScalarType;
  typedef  short           OutputPixelType;


  typedef typename itk::Image< PixelType, Dimension >  InputImageType;
  typedef typename itk::Image< OutputPixelType , Dimension >  OutputImageType;

  typename InputImageType::Pointer imFixed = 0;
  typename InputImageType::Pointer imMoving = 0;

  typedef typename itk::ImageFileReader< InputImageType  > FixedImageReaderType;
  typedef typename itk::ImageFileReader< InputImageType >  MovingImageReaderType;
  typedef typename itk::ImageFileWriter< OutputImageType > OutputImageWriterType;
  
  typedef typename itk::ImageMomentsCalculator< InputImageType > ImageMomentCalculatorType;
  typedef typename ImageMomentCalculatorType::AffineTransformType AffineTransformType;
 

  typedef itk::CropTargetImageWhereSourceImageNonZeroImageFilter< InputImageType, InputImageType > MaskFilterType;


  // Load both images to be registered
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  try 
  { 
    // The fixed image

    typename FixedImageReaderType::Pointer  fixedImageReader  = FixedImageReaderType::New();

    fixedImageReader->SetFileName(  args.fileFixedImage );

    std::cout << "Loading fixed image: " << args.fileFixedImage<< std::endl;
    fixedImageReader->Update();
      
    imFixed = fixedImageReader->GetOutput();
    imFixed->DisconnectPipeline();
    fixedImageReader = 0;

    // The fixed image mask

    if (args.fileFixedMask.length() > 0)
    {
      typename FixedImageReaderType::Pointer  fixedMaskReader  = FixedImageReaderType::New();

      fixedMaskReader->SetFileName(   args.fileFixedMask );

      std::cout << "Loading fixed mask: " << args.fileFixedMask<< std::endl;
      fixedMaskReader->Update();  

      typename MaskFilterType::Pointer maskFilter = MaskFilterType::New();  
  
      maskFilter->SetInput1( fixedMaskReader->GetOutput() );
      maskFilter->SetInput2( imFixed );

      maskFilter->Update();  

      imFixed = maskFilter->GetOutput();
      imFixed->DisconnectPipeline();
      maskFilter = 0;      
    }

    // The moving image

    typename MovingImageReaderType::Pointer movingImageReader = MovingImageReaderType::New();

    movingImageReader->SetFileName( args.fileMovingImage );

    std::cout << "Loading moving image: " << args.fileMovingImage<< std::endl;
    movingImageReader->Update();
         
    imMoving = movingImageReader->GetOutput();
    imMoving->DisconnectPipeline();
    movingImageReader = 0;

    // The moving image mask
         
    if (args.fileMovingMask.length() > 0)
    {
      typename MovingImageReaderType::Pointer movingMaskReader = MovingImageReaderType::New();

      movingMaskReader->SetFileName(  args.fileMovingMask );

      std::cout << "Loading moving mask: " << args.fileMovingMask<< std::endl;
      movingMaskReader->Update();  

      typename MaskFilterType::Pointer maskFilter = MaskFilterType::New();  
  
      maskFilter->SetInput1( movingMaskReader->GetOutput() );
      maskFilter->SetInput2( imMoving );

      maskFilter->Update();  

      imMoving = maskFilter->GetOutput();
      imMoving->DisconnectPipeline();
      maskFilter = 0;      
    }
  } 
  catch( itk::ExceptionObject & err ) 
  { 
    std::cerr <<"ExceptionObject caught !";
    std::cerr << err << std::endl; 
    return -2;
  }                


  // Compute the  registration matrix
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  // Compute the fixed image physical axes to principal axes transform

  typename ImageMomentCalculatorType::Pointer 
    fixedImageMomentCalculator = ImageMomentCalculatorType::New(); 

  fixedImageMomentCalculator->SetImage( imFixed ); 
  fixedImageMomentCalculator->Compute(); 
  
  typename AffineTransformType::Pointer 
    fixedImageTransform = fixedImageMomentCalculator->GetPhysicalAxesToPrincipalAxesTransform();


  // Compute the moving image principal axes to physical axes transform

  typename ImageMomentCalculatorType::Pointer
    movingImageMomentCalculator = ImageMomentCalculatorType::New(); 

  movingImageMomentCalculator->SetImage( imMoving ); 
  movingImageMomentCalculator->Compute(); 
  
  typename AffineTransformType::Pointer 
    movingImageTransform = movingImageMomentCalculator->GetPrincipalAxesToPhysicalAxesTransform();


  // Compute the scale factors in 'x' and 'y' from the normalised principal moments

  typename ImageMomentCalculatorType::ScalarType 
    fixedTotalMass = fixedImageMomentCalculator->GetTotalMass();

  typename ImageMomentCalculatorType::VectorType 
    fixedPrincipalMoments = fixedImageMomentCalculator->GetPrincipalMoments();

  typename ImageMomentCalculatorType::ScalarType 
    movingTotalMass = movingImageMomentCalculator->GetTotalMass();

  typename ImageMomentCalculatorType::VectorType 
    movingPrincipalMoments = movingImageMomentCalculator->GetPrincipalMoments(); 
  
  itk::FixedArray< double, Dimension > scaleFactor;

  for ( unsigned int iDim; iDim<Dimension; iDim++ )
  {
    scaleFactor[ iDim ] = 
      sqrt(movingPrincipalMoments[ iDim ] / movingTotalMass )
      / sqrt( fixedPrincipalMoments[ iDim ] / fixedTotalMass );
  }

  std::cout << "Scale factors: " << scaleFactor << std::endl;

  typedef itk::ScaleTransform< double, Dimension > ScaleTransformType;
  typename ScaleTransformType::Pointer scaleTransform = ScaleTransformType::New();

  scaleTransform->SetScale( scaleFactor );

  movingImageTransform->Compose( scaleTransform, true );
  movingImageTransform->Compose( fixedImageTransform, true );


  // Resample the moving image
  // ~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef typename itk::ResampleImageFilter<InputImageType, OutputImageType> ResampleFilterType;
  
  typename ResampleFilterType::Pointer resampleFilter = ResampleFilterType::New();

  typedef typename itk::LinearInterpolateImageFunction< InputImageType, double > InterpolatorType;

  typename InterpolatorType::Pointer interpolator = InterpolatorType::New();

  resampleFilter->SetInput( imMoving );

  resampleFilter->SetTransform( movingImageTransform );

  resampleFilter->SetInterpolator( interpolator );
  resampleFilter->SetDefaultPixelValue( 0 );

  resampleFilter->SetOutputOrigin( imFixed->GetOrigin() );
  resampleFilter->SetOutputSpacing( imFixed->GetSpacing() );
  resampleFilter->SetOutputDirection( imFixed->GetDirection() );
  resampleFilter->SetSize( imFixed->GetLargestPossibleRegion().GetSize() );
  
  try 
  { 
    resampleFilter->Update();             
  }
  catch( itk::ExceptionObject & excp )
  {
    std::cerr << "Exception caught: " << std::endl;
    std::cerr << excp << std::endl;
    return EXIT_FAILURE;
  }


  // Write the transformed output image
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if (args.fileOutputImage.length() > 0)
  {
    typename OutputImageWriterType::Pointer imageWriter = OutputImageWriterType::New();

    imageWriter->SetFileName( args.fileOutputImage );
    imageWriter->SetInput( resampleFilter->GetOutput() );
  
    try
    {
      std::cout << "Writing transformed moving image to: " << args.fileOutputImage << std::endl; 
      imageWriter->Update(); 
    }
    catch( itk::ExceptionObject & err ) 
    { 
      std::cerr << "ERROR: Failed to mask and write the image to a file." << err << std::endl; 
      return EXIT_FAILURE;
    }                
  }
    
  // Save the transform
  // ~~~~~~~~~~~~~~~~~~

  if (args.fileOutputTransformFile.length() > 0)
  {
    typedef typename itk::TransformFileWriter TransformFileWriterType;
    typename TransformFileWriterType::Pointer transformFileWriter = TransformFileWriterType::New();
  
    transformFileWriter->SetInput( movingImageTransform );
    transformFileWriter->SetFileName( args.fileOutputTransformFile);

    try 
    { 
      transformFileWriter->Update();             
    }
    catch( itk::ExceptionObject & excp )
    {
      std::cerr << "Exception caught: " << std::endl;
      std::cerr << excp << std::endl;
      return EXIT_FAILURE;
    }
  }

  return EXIT_SUCCESS;
}


/**
 * \brief Performs a image moments  registration
 */
int main(int argc, char** argv)
{
  // To pass around command line args
  PARSE_ARGS;

  // To pass around command line args
  struct arguments args;


  args.fileFixedImage  = fileFixedImage;
  args.fileMovingImage = fileMovingImage;

  args.fileOutputImage = fileOutputImage;

  args.fileOutputTransformFile = fileOutputTransformFile;

  args.fileFixedMask  = fileFixedMask;
  args.fileMovingMask = fileMovingMask;

  // Print out the options
  
  std::cout << std::endl
            << "Command line options: "			<< std::endl;

  std::cout << "  Mandatory Input and Output Options: "	<< std::endl
            << "    Fixed target image: "		<< args.fileFixedImage          << std::endl
            << "    Moving source image: "		<< args.fileMovingImage         << std::endl
            << "    Output  transformation: "	<< args.fileOutputTransformFile << std::endl
            << "    Output registered image: "		<< args.fileOutputImage         << std::endl;

  std::cout << "  Common Options: "			<< std::endl
            << "    Fixed target mask image: "		<< args.fileFixedMask           << std::endl
            << "    Moving source mask image: "		<< args.fileMovingMask          << std::endl
            << std::endl;


  // Validation
  if ( ( args.fileFixedImage.length() <= 0 )|| 
       ( args.fileMovingImage.length() <= 0 ) || 
       ( args.fileOutputTransformFile.length() <= 0 ) )
  {
    commandLine.getOutput()->usage(commandLine);
    std::cout << std::endl << "  -help for more options" << std::endl << std::endl;
    return -1;
  }

  unsigned int dims = itk::PeekAtImageDimensionFromSizeInVoxels( args.fileFixedImage );
  if (dims != 3 && dims != 2)
  {
    std::cout << "Unsupported image dimension" << std::endl;
    return EXIT_FAILURE;
  }

  int result;

  switch ( dims )
  {
  case 2:
    result = DoMain<2>(args);
    break;
  case 3:
    result = DoMain<3>(args);
    break;
  default:
    std::cout << "Unsupported image dimension" << std::endl;
    exit( EXIT_FAILURE );
  }
  return result;
}
