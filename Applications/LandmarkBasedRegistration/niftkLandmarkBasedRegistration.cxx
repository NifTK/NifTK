/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <fstream>
#include <iomanip>

#include <niftkConversionUtils.h>
#include <itkCommandLineHelper.h>
#include <itkDOMReader.h>

#include <itkImageFileWriter.h>
#include <itkImageFileReader.h>
#include <itkImage.h>
#include <itkResampleImageFilter.h>
#include <niftkCSVRow.h>
#include <itkTransformFileWriter.h>
#include <itkTransformFileReader.h>
#include <itkTransformFactory.h>
#include <itkLandmarkBasedTransformInitializer.h>
#include <itkRigid2DTransform.h>
 
#include <niftkLandmarkBasedRegistrationCLP.h>


/*!
 * \file niftkLandmarkBasedRegistration.cxx
 * \page niftkLandmarkBasedRegistration
 * \section niftkLandmarkBasedRegistrationSummary Transforms an image using itk::LandmarkBasedTransformInitializer. The transform computed gives the best fit transform that maps the fixed and moving images in a least squares sense.
 */



struct arguments
{
  bool flgVerbose;
  bool flgDebug;

  std::string fileInputSourceImage;
  std::string fileInputTargetImage;

  std::string fileSourceLandmarks;
  std::string fileTargetLandmarks;

  std::string fileOutputImage;
  std::string fileOutputTransformation;
  std::string fileOutputDeformationField;

  arguments() {
    flgVerbose = false;
    flgDebug = false;
  }
};



// --------------------------------------------------------------------------------
// Read an csv point set
// --------------------------------------------------------------------------------

template <class PointType, class PointSetType>
void ReadCSVPointSet( std::string fileInput, 
                      bool flgVerbose,
                      PointSetType &PointSet )
{
  unsigned int iLandmark = 0;

  PointType point;

  if ( fileInput.length() )
  {
 
    std::ifstream fin( fileInput.c_str(), std::ifstream::in|std::ifstream::binary );

    if ( (! fin) || fin.bad() || (! fin.is_open()) ) 
    {
      std::cerr << "ERROR: Could not open file: " << fileInput << std::endl;
      return;
    }
              
    std::cout << std::endl << "Reading CSV file: " << fileInput << std::endl;

    niftk::CSVRow csvRow;
              
    while ( true )
    {
      fin >> csvRow;

      if ( fin.eof() )
      {
        if ( flgVerbose )
        {
          std::cout << "End of file reached" << std::endl;
        }
        break;
      }

      for (unsigned int i=0; i < point.GetPointDimension(); i++)
      {
        point[i] = atof( csvRow[i].c_str() );
      }
      
      if ( flgVerbose )
      {
        std::cout << std::setw(8) << iLandmark << " "
                  << point << std::endl;
      }

      PointSet.push_back( point );
      iLandmark++;
    }
  }
};



// --------------------------------------------------------------------------------
// Read an MITK point set
// --------------------------------------------------------------------------------

template <class PointType, class PointSetType>
void ReadMITKPointSet( std::string fileInput, 
                       bool flgVerbose,
                       PointSetType &PointSet )
{
  unsigned int iLandmark = 0;

  PointType point;

  if ( fileInput.length() )
  {
    typename itk::DOMNodeXMLReader::Pointer domReader = itk::DOMNodeXMLReader::New();

    domReader->SetFileName( fileInput );

    try {
      std::cout << "Is point set: " << fileInput << " MITK?";
      domReader->Update();
    }
    catch( itk::ExceptionObject & err )
    {
      std::cout << " - No" << std::endl;
      return;
    }
    
    typename itk::DOMNode::Pointer dom = domReader->GetOutput();

    typedef itk::DOMNode::IdentifierType NodeIdentifierType;

    NodeIdentifierType iNodePoint;

    typename itk::DOMNode::Pointer nodePointSet;
    typename itk::DOMNode::Pointer nodePoint;

    const itk::DOMTextNode* textNode;

    nodePointSet = dom->Find( "point_set" );

    typedef itk::DOMNode::ConstChildrenListType ConstChildrenListType;
    typedef itk::DOMNode::AttributesListType AttributesListType;

    ConstChildrenListType children;
    nodePointSet->GetAllChildren( children );

    for ( size_t i = 0; i < children.size(); i++ )
    {

      ConstChildrenListType pointChildren;
      children[i]->GetAllChildren( pointChildren );

      for ( size_t i = 0; i < pointChildren.size(); i++ )
      {
        if ( pointChildren[i]->GetName() == std::string( "point" ) )
        {
          
          ConstChildrenListType coords;
          pointChildren[i]->GetAllChildren( coords );

          int iPoint = atoi( coords[0]->GetTextChild()->GetText().c_str() );

          for ( unsigned int iDim=0; iDim<point.GetPointDimension(); iDim++ )
          {
            point[ iDim ] = atof( coords[iDim + 2]->GetTextChild()->GetText().c_str() );
          }

          if ( flgVerbose )
          {
            std::cout << std::setw(8) << iLandmark << " "
                      << point << std::endl;
          }

          PointSet.push_back( point );
          iLandmark++;
        }
      }
    }
  }
}


// --------------------------------------------------------------------------------
// StringEndsWith( std::string const &fullString, std::string const &suffix ) 
// --------------------------------------------------------------------------------

bool StringEndsWith( std::string const &fullString, 
                     std::string const &suffix ) 
{
  if ( fullString.length() >= suffix.length() ) 
  {
    return ( 0 == fullString.compare( fullString.length() - suffix.length(), 
                                      suffix.length(), suffix ) );
  } 

  return false;
}


// --------------------------------------------------------------------------------
// DoMain(arguments args)
// --------------------------------------------------------------------------------

template <int ImageDimension, class PixelType>
int DoMain(arguments args)
{
  typedef itk::Image< PixelType, ImageDimension >          InputImageType;

  typedef itk::ImageFileReader< InputImageType  >          ReaderType;
  typedef itk::ImageFileWriter< InputImageType >           DeformedImageWriterType;

  typedef itk::Point<  float, ImageDimension >             FieldPointType;
  typedef itk::Vector< float, ImageDimension >             FieldVectorType;

  typedef itk::Image< FieldVectorType,  ImageDimension >   DisplacementFieldType;

  typedef itk::ImageFileWriter< DisplacementFieldType >    FieldWriterType;

  typedef double CoordinateRepType;

  typedef itk::ResampleImageFilter< InputImageType, InputImageType  > ResamplerType;

  typedef itk::LinearInterpolateImageFunction< InputImageType, CoordinateRepType > InterpolatorType;

  typedef itk::AffineTransform< CoordinateRepType, ImageDimension > TransformType;

  typedef itk::LandmarkBasedTransformInitializer< TransformType, InputImageType, InputImageType > 
      LandmarkBasedTransformInitializerType;

  //  Create source and target landmarks.
  typedef typename LandmarkBasedTransformInitializerType::LandmarkPointContainer PointSetType;
  typedef typename LandmarkBasedTransformInitializerType::LandmarkPointType      PointType;
 


  // Read the input points
  // ~~~~~~~~~~~~~~~~~~~~~

  PointSetType sourceLandMarks;
  PointSetType targetLandMarks;

  std::string csvSuffix( ".csv" );
  std::string mitkSuffix( ".mitk" );

  // MITK Point Set?

  if ( StringEndsWith( args.fileSourceLandmarks, mitkSuffix ) )
  {
    ReadMITKPointSet<PointType, PointSetType>( args.fileSourceLandmarks, 
                                               args.flgVerbose, 
                                               sourceLandMarks);
  }

  if ( StringEndsWith( args.fileTargetLandmarks, mitkSuffix ) )
  {
    ReadMITKPointSet<PointType, PointSetType>( args.fileTargetLandmarks, 
                                               args.flgVerbose, 
                                               targetLandMarks );
  }
    
  // CSV File?
    
  if ( ( ! sourceLandMarks.size() ) && StringEndsWith( args.fileSourceLandmarks, csvSuffix ) )
  {
    ReadCSVPointSet<PointType, PointSetType>( args.fileSourceLandmarks, 
                                              args.flgVerbose,
                                              sourceLandMarks );
  }
  else 
  {
    std::cerr << "WARNING: Source landmarks file: " << args.fileSourceLandmarks 
              << " does not end in '.csv' or '.mitk'." << std::endl;
  }
 
  if ( ( ! targetLandMarks.size() ) && StringEndsWith( args.fileTargetLandmarks, csvSuffix ) )
  {
    ReadCSVPointSet<PointType, PointSetType>( args.fileTargetLandmarks, 
                                              args.flgVerbose, 
                                              targetLandMarks );
  }
  else 
  {
    std::cerr << "WARNING: Target landmarks file: " << args.fileTargetLandmarks 
              << " does not end in '.csv' or '.mitk'." << std::endl;
  }



  if ( ! targetLandMarks.size() )
  {
    std::cerr << "ERROR: Failed to read target landmarks" << std::endl;
    return EXIT_FAILURE;
  }

  if ( ! sourceLandMarks.size() )
  {
    std::cerr << "ERROR: Failed to read source landmarks" << std::endl;
    return EXIT_FAILURE;
  }

  if ( targetLandMarks.size() != sourceLandMarks.size() )
  {
    std::cerr << "ERROR: Numbers of target (" << targetLandMarks.size() 
              << ") and source (" << sourceLandMarks.size() << ") landmarks differ" << std::endl;
    return EXIT_FAILURE;
  }



  // Compute the transformation
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~
    
  typename LandmarkBasedTransformInitializerType::Pointer landmarkBasedTransformInitializer =
    LandmarkBasedTransformInitializerType::New();


  landmarkBasedTransformInitializer->SetFixedLandmarks( targetLandMarks );
  landmarkBasedTransformInitializer->SetMovingLandmarks( sourceLandMarks );
    
  typename TransformType::Pointer transform = TransformType::New();
 
  transform->SetIdentity();

  landmarkBasedTransformInitializer->SetTransform( transform );
  landmarkBasedTransformInitializer->InitializeTransform();


  // Save the transformation to a file?
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( args.fileOutputTransformation.length() )
  {
    typedef itk::TransformFileWriterTemplate< CoordinateRepType > TransformWriterType;
    typename TransformWriterType::Pointer transformWriter = TransformWriterType::New();
      
    transformWriter->SetInput( transform );
      
    transformWriter->SetFileName( args.fileOutputTransformation );
      
    try
    {
      std::cout << "Writing the transformation to file: " 
                << args.fileOutputTransformation << std::endl;
      transformWriter->Update();       
    }
    catch( itk::ExceptionObject & excp )
    {
      std::cerr << "ERROR: Failed to write the transformation, exception: " << std::endl
                << excp << std::endl;
      return EXIT_FAILURE;
    }
  }


  // Read the input source image
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typename ReaderType::Pointer sourceReader = ReaderType::New();

  sourceReader->SetFileName( args.fileInputSourceImage );

  try
  {
    std::cout << "Reading input source image: " << args.fileInputSourceImage << std::endl;
    sourceReader->Update(); 
  }
  catch( itk::ExceptionObject & err ) 
  { 
    std::cerr << "ERROR: Failed to read image, exception: " << std::endl
              << err << std::endl; 
    return EXIT_FAILURE;
  }                

  typename InputImageType::ConstPointer inputImage = sourceReader->GetOutput();


  // Read the input target image
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typename InputImageType::ConstPointer targetImage;

  if ( args.fileInputTargetImage.length() )
  {

    typename ReaderType::Pointer targetReader = ReaderType::New();

    targetReader->SetFileName( args.fileInputTargetImage );

    try
    {
      std::cout << "Reading input target image: " << args.fileInputTargetImage << std::endl;
      targetReader->Update(); 
    }
    catch( itk::ExceptionObject & err ) 
    { 
      std::cerr << "ERROR: Failed to read image, exception: " << std::endl
                << err << std::endl; 
      return EXIT_FAILURE;
    }                
    
    targetImage = targetReader->GetOutput();
  }

  else
  {
    targetImage = inputImage;
  }


  // The image is then resampled to produce an output image as defined by the
  // transform. Here we use a LinearInterpolator.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typename ResamplerType::Pointer resampler = ResamplerType::New();
  typename InterpolatorType::Pointer interpolator = InterpolatorType::New();

  resampler->SetInterpolator( interpolator );

  typename InputImageType::SpacingType spacing      = targetImage->GetSpacing();
  typename InputImageType::PointType   origin       = targetImage->GetOrigin();
  typename InputImageType::DirectionType direction  = targetImage->GetDirection();
  typename InputImageType::RegionType region        = targetImage->GetBufferedRegion();
  typename InputImageType::SizeType   size          = region.GetSize();

  resampler->SetOutputSpacing( spacing );
  resampler->SetOutputDirection( direction );
  resampler->SetOutputOrigin(  origin  );
  resampler->SetSize( size );
  resampler->SetTransform( transform );

  resampler->SetOutputStartIndex(  region.GetIndex() );
  resampler->SetInput( inputImage );

  try
  {
    std::cout << "Transforming the input image" << std::endl;
    resampler->Update();
  }
  catch( itk::ExceptionObject & excp )
  {
    std::cerr << "ERROR: Cannot transform the image, exception thrown " << std::endl
              << excp << std::endl;
    return EXIT_FAILURE;
  }


  // Write the deformed image
  // ~~~~~~~~~~~~~~~~~~~~~~~~

  if ( args.fileOutputImage.length() )
  {

    typename DeformedImageWriterType::Pointer deformedImageWriter = DeformedImageWriterType::New();

    deformedImageWriter->SetInput( resampler->GetOutput() );
    deformedImageWriter->SetFileName( args.fileOutputImage );

    try
    {
      std::cout << "Writing the deformed image to file: " 
                << args.fileOutputImage << std::endl;
      deformedImageWriter->Update();
    }
    catch( itk::ExceptionObject & excp )
    {
      std::cerr << "ERROR: Cannot write the transformed image to a file, exception thrown " << std::endl
                << excp << std::endl;
      return EXIT_FAILURE;
    }
  }


  // Compute the deformation field

  // The deformation field is computed as the difference between the
  // input and the deformed image by using an iterator.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( args.fileOutputDeformationField.length() )
  {

    typename DisplacementFieldType::Pointer field = DisplacementFieldType::New();

    field->SetRegions( region );
    field->SetOrigin( origin );
    field->SetSpacing( spacing );
    field->Allocate();

    typedef itk::ImageRegionIterator< DisplacementFieldType > FieldIterator;

    FieldIterator fi( field, region );
    fi.GoToBegin();

    typename TransformType::InputPointType  point1;
    typename TransformType::OutputPointType point2;
    typename DisplacementFieldType::IndexType index;

    FieldVectorType displacement;

    while( ! fi.IsAtEnd() )
    {
      index = fi.GetIndex();

      field->TransformIndexToPhysicalPoint( index, point1 );

      point2 = transform->TransformPoint( point1 );

      for ( unsigned int i = 0;i < ImageDimension;i++)
      {
        displacement[i] = point2[i] - point1[i];
      }
      fi.Set( displacement );

      ++fi;
    }


    // Write computed deformation field

    typename FieldWriterType::Pointer fieldWriter = FieldWriterType::New();
  
    fieldWriter->SetFileName( args.fileOutputDeformationField );

    fieldWriter->SetInput( field );

    try
    {
      std::cout << "Writing the deformation field to file: " 
                << args.fileOutputDeformationField << std::endl;
      fieldWriter->Update();
    }

    catch( itk::ExceptionObject &excp )
    {
      std::cerr << "ERROR: Failed to write the deformation field, exception thrown " << std::endl;
      std::cerr << excp << std::endl;
      return EXIT_FAILURE;
    }
  }

  return EXIT_SUCCESS;
}




/**
 * \brief Transforms an image using a transformation computed from a pair of point sets.
 */

int main(int argc, char** argv)
{
  unsigned int i;

  int result;


  struct arguments args;

  PARSE_ARGS;

  args.flgVerbose = flgVerbose;
  args.flgDebug   = flgDebug;

  args.fileOutputImage            = fileOutputImage;
  args.fileOutputDeformationField = fileOutputDeformationField;
  args.fileOutputTransformation   = fileOutputTransform;

  args.fileSourceLandmarks = fileSourceLandmarks;
  args.fileTargetLandmarks = fileTargetLandmarks;
  
  args.fileInputSourceImage = fileInputSourceImage;
  args.fileInputTargetImage = fileInputTargetImage;


  if ( ! ( args.fileSourceLandmarks.length() || 
           args.fileTargetLandmarks.length() ) )
  {
    std::cerr << "ERROR: Two point sets must be specified" << std::endl;
    return EXIT_FAILURE;
  }

  if ( ! args.fileInputSourceImage.length() )
  {
    std::cerr << "ERROR: An input image must be specified" << std::endl;
    return EXIT_FAILURE;
  }

  int dims = itk::PeekAtImageDimensionFromSizeInVoxels(args.fileInputTargetImage);
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
   
  itk::ImageIOBase::Pointer imageIO = 
    itk::ImageIOFactory::CreateImageIO( args.fileInputTargetImage.c_str(), 
                                        itk::ImageIOFactory::ReadMode );

  imageIO->SetFileName( args.fileInputTargetImage );
  imageIO->ReadImageInformation();

  itk::ImageIOBase::IOPixelType PixelType = imageIO->GetPixelType();

  std::cout << "Dimensions        :\t" << dims << std::endl
            << "PixelType         :\t" << imageIO->GetPixelTypeAsString( PixelType ) << std::endl;
            

  switch ( PixelType )
  {
  case itk::ImageIOBase::SCALAR:
  {
 
  switch (itk::PeekAtComponentType(args.fileInputTargetImage))
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
   
    break;
  }
  
  case itk::ImageIOBase::RGB:
  {
  
    switch ( imageIO->GetComponentType() )
    {
    case itk::ImageIOBase::UCHAR:
      std::cout << "Image is RGB unsigned char" << std::endl;
      if (dims == 2)
      {
        return DoMain<2, itk::RGBPixel<unsigned char> >( args );
      }
      else
      {
        return DoMain<3, itk::RGBPixel<unsigned char> >( args );
      }
      break;
    case itk::ImageIOBase::SHORT:
      std::cout << "Image is RGB short" << std::endl;
      if (dims == 2)
      {
        return DoMain<2, itk::RGBPixel<short> >( args );
      }
      else
      {
        return DoMain<3, itk::RGBPixel<short> >( args );
      }
      break;
    default:
      std::cerr << "ERROR: Non standard RGB component type: " 
                << imageIO->GetComponentTypeAsString(imageIO->GetComponentType()) << std::endl;
      return EXIT_FAILURE;
    }

    break;
  }

  default:
    std::cerr << "ERROR: Non standard pixel format" << std::endl;
    return EXIT_FAILURE;
  }

  return result;
}
