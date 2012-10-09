/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-07-22 10:11:40 +0100 (Thu, 22 Jul 2010) $
 Revision          : $Revision: 3539 $
 Last modified by  : $Author: kkl $

 Original author   : leung@drc.ion.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "ConversionUtils.h"
#include "CommandLineParser.h"
#include "itkCommandLineHelper.h"

#include "itkAffineTransform.h"
#include "itkTransformFileReader.h"

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include "itkResampleImageFilter.h"


struct niftk::CommandLineArgumentDescription clArgList[] = {

  {OPT_SWITCH, "inv",  NULL, "Invert the global affine transformation."},
  {OPT_STRING|OPT_REQ, "g",  "filename", "Input global affine transformation."},

  {OPT_STRING, "ti",  "filename", 
   "Input fixed/target image which defines the image region to use \n"
   "(if not specified then the source image space is used)."},

  {OPT_STRING|OPT_REQ, "si",  "filename", 
   "Input source/moving image to be transformed."},
  
  {OPT_STRING|OPT_REQ, "o", "filename", "Output transformed image."},
  
  {OPT_DONE, NULL, NULL, 
   "Transform an image using an ITK affine transformation "
   "(see also: niftkTransformation)."} 
};


enum {
  O_INVERT_TRANSFORM,
  O_INPUT_TRANSFORM,

  O_INPUT_TARGET, 
  O_INPUT_SOURCE, 
  
  O_OUTPUT_FILE
};

struct arguments
{
  bool flgInvert;
  std::string fileTransform;

  std::string fileTarget;
  std::string fileSource;

  std::string fileOutput; 
};


template <int Dimension>
int DoMain(arguments args)
{
  typedef short PixelType;                                          
  typedef itk::Image< PixelType, Dimension > ImageType;

  typename ImageType::Pointer fixedImage = 0;
  typename ImageType::Pointer movingImage = 0;
  
  typedef itk::ImageFileReader< ImageType  > ImageReaderType;

  typedef itk::AffineTransform<double, Dimension> AffineTransformType; 

  typename itk::TransformFileReader::TransformListType* transforms = 0;
  
  typename AffineTransformType::Pointer inputTransform = 0; 
  typename AffineTransformType::Pointer inverseTransform = 0; 

  typename itk::TransformFileReader::TransformListType::const_iterator iterTransforms;


  try
  {
    typedef itk::TransformFileReader TransformFileReaderType;
    TransformFileReaderType::Pointer transformFileReader = TransformFileReaderType::New();


    // Read the transformation

    transformFileReader->SetFileName( args.fileTransform );
    transformFileReader->Update();

    transforms = transformFileReader->GetTransformList();
    std::cout << "Reading transform from file: " << args.fileTransform.c_str() << std::endl
	      << "Number of transforms = " << transforms->size() << std::endl;

    iterTransforms = transforms->begin();

    inputTransform = static_cast<AffineTransformType*>( (*iterTransforms).GetPointer() );
    inputTransform->Print( std::cout );


    // Invert the transformation?

    if ( args.flgInvert )
    {
      inverseTransform = AffineTransformType::New();

      inverseTransform->SetCenter( inputTransform->GetCenter() );
      inputTransform->GetInverse( inverseTransform );
      inverseTransform->Print( std::cout );

      inputTransform = inverseTransform;
    }


    // Read the input images

    typename ImageReaderType::Pointer movingImageReader = ImageReaderType::New();
    movingImageReader->SetFileName( args.fileSource );
    movingImageReader->Update();
    movingImage = movingImageReader->GetOutput();
      
    if ( args.fileTarget.length() > 0 )
    {
      typename ImageReaderType::Pointer fixedImageReader  = ImageReaderType::New();
      fixedImageReader->SetFileName( args.fileTarget );
      fixedImageReader->Update();
      fixedImage = fixedImageReader->GetOutput();
    }

    
    // Transform the image

    typedef itk::ResampleImageFilter< ImageType, ImageType > ResampleFilterType;

    typename ResampleFilterType::Pointer resampler = ResampleFilterType::New();

    resampler->SetInput( movingImage );
    resampler->SetTransform( inputTransform );
 
    if ( fixedImage ) 
    {
      resampler->SetSize( fixedImage->GetLargestPossibleRegion().GetSize() );
      resampler->SetOutputOrigin(  fixedImage->GetOrigin() );
      resampler->SetOutputSpacing( fixedImage->GetSpacing() );
      resampler->SetOutputDirection( fixedImage->GetDirection() );
    }
    else 
    {
      resampler->SetSize( movingImage->GetLargestPossibleRegion().GetSize() );
      resampler->SetOutputOrigin(  movingImage->GetOrigin() );
      resampler->SetOutputSpacing( movingImage->GetSpacing() );
      resampler->SetOutputDirection( movingImage->GetDirection() );
    }

    resampler->SetDefaultPixelValue( 0 );

    resampler->Update();


    // Write the resampled image to a file

    typedef itk::ImageFileWriter< ImageType >  WriterType;

    typename WriterType::Pointer writer =  WriterType::New();

    writer->SetFileName( args.fileOutput.c_str() );
    writer->SetInput( resampler->GetOutput() );

    writer->Update();

  }  
  catch (itk::ExceptionObject& exceptionObject)
  {
    std::cerr << "ERROR: Failed compose tranformations:" << exceptionObject << std::endl;
    return EXIT_FAILURE; 
  }
  
  return EXIT_SUCCESS; 
}

  

int main(int argc, char** argv)
{
  struct arguments args;

  niftk::CommandLineParser CommandLineOptions( argc, argv, clArgList, true );
  
  CommandLineOptions.GetArgument( O_INVERT_TRANSFORM, args.flgInvert );
  CommandLineOptions.GetArgument( O_INPUT_TRANSFORM,  args.fileTransform );
  
  CommandLineOptions.GetArgument( O_INPUT_TARGET, args.fileTarget );
  CommandLineOptions.GetArgument( O_INPUT_SOURCE, args.fileSource );

  CommandLineOptions.GetArgument( O_OUTPUT_FILE, args.fileOutput );

  unsigned int dims = itk::PeekAtImageDimensionFromSizeInVoxels(args.fileSource);

  if (dims != 3 && dims != 2)
  {
    std::cout << "Unsupported image dimension" << std::endl;
    return EXIT_FAILURE;
  }

  int result;

  switch ( dims )
  {
  case 2:
    std::cout << "Image is 2D" << std::endl;
    result = DoMain<2>( args );
    break;
  case 3:
    std::cout << "Image is 3D" << std::endl;
    result = DoMain<3>( args );
    break;
  default:
    std::cout << "ERROR: Unsupported image dimension" << std::endl;
    exit( EXIT_FAILURE );
  }
  return result;
}
  
