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

#include <itkLogHelper.h>

#include <niftkConversionUtils.h>
#include <itkCommandLineHelper.h>
#include <itkDOMReader.h>

#include <itkImageFileWriter.h>
#include <itkImageFileReader.h>
#include <itkImage.h>
#include <itkResampleImageFilter.h>
#include <itkElasticBodySplineKernelTransform.h>
#include <niftkCSVRow.h>
#include <itkTransformFileWriter.h>
#include <itkTransformFileReader.h>
#include <itkTransformFactory.h>

#include <niftkElasticBodySplineWarpCLP.h>


/*!
 * \file niftkElasticBodySplineWarp.cxx
 * \page niftkElasticBodySplineWarp
 * \section niftkElasticBodySplineWarpSummary Computes an elastic body spline warp from a set of landmarks.
 */



struct arguments
{
  bool flgVerbose;
  bool flgDebug;

  std::string fileInputSourceImage;
  std::string fileInputTargetImage;

  std::string fileSourceLandmarks;
  std::string fileTargetLandmarks;

  std::string fileInputElasticBodySplineMatrix;

  std::string fileOutputImage;
  std::string fileOutputElasticBodySplineMatrix;
  std::string fileOutputDeformationField;

  float poisson;
  float stiffness;

  arguments() {
    flgVerbose = false;
    flgDebug = false;

    poisson = 0.49;
    stiffness = 0.;
  }
};



// --------------------------------------------------------------------------------
// Read an csv point set
// --------------------------------------------------------------------------------

template <class PointType, class PointSetType>
typename PointSetType::Pointer ReadCSVPointSet( std::string fileInput, bool flgVerbose )
{
  unsigned int iLandmark = 0;

  typename PointSetType::Pointer PointSet = PointSetType::New();

  PointType point;

  if ( fileInput.length() )
  {
 
    std::ifstream fin( fileInput.c_str() );

    if ((! fin) || fin.bad()) 
    {
      std::cerr << "ERROR: Could not open file: " << fileInput << std::endl;
      return 0;
    }
              
    std::cout << std::endl << "Reading CSV file: " << fileInput << std::endl;

    niftk::CSVRow csvRow;
              
    while( fin >> csvRow )
    {
      for (unsigned int i=0; i < point.Size(); i++)
      {
        point[i] = atof( csvRow[i].c_str() );
      }
      
      if ( flgVerbose )
      {
        std::cout << std::setw(8) << iLandmark << " "
                  << point << std::endl;
      }

      PointSet->SetPoint( iLandmark, point );
      iLandmark++;
    }
  }

  return PointSet;

};



// --------------------------------------------------------------------------------
// Read an MITK point set
// --------------------------------------------------------------------------------

template <class PointType, class PointSetType>
typename PointSetType::Pointer ReadMITKPointSet( std::string fileInput, bool flgVerbose )
{
  unsigned int iLandmark = 0;

  typename PointSetType::Pointer PointSet = PointSetType::New();

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
      return 0;
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

          point[0] = atof( coords[2]->GetTextChild()->GetText().c_str() );
          point[1] = atof( coords[3]->GetTextChild()->GetText().c_str() );
          point[2] = atof( coords[4]->GetTextChild()->GetText().c_str() );

          if ( flgVerbose )
          {
            std::cout << std::setw(8) << iLandmark << " "
                      << point << std::endl;
          }

          PointSet->SetPoint( iLandmark, point );
          iLandmark++;
        }
      }
    }
  }

  return PointSet;
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

  typedef itk::Point< CoordinateRepType, ImageDimension >  PointType;
  typedef std::vector< PointType >                         PointArrayType;

  typedef itk::ElasticBodySplineKernelTransform< CoordinateRepType, ImageDimension> ElasticBodyTransformType;

  typedef typename ElasticBodyTransformType::PointSetType PointSetType;
  
  typedef typename PointSetType::Pointer PointSetPointer;
  typedef typename PointSetType::PointIdentifier PointIdType;


  typedef itk::ResampleImageFilter< InputImageType, InputImageType  > ResamplerType;

  typedef itk::LinearInterpolateImageFunction< InputImageType, double > InterpolatorType;


  // Read the input points
  // ~~~~~~~~~~~~~~~~~~~~~

  typename ElasticBodyTransformType::Pointer ebs;

  if ( args.fileSourceLandmarks.length() && args.fileTargetLandmarks.length() )
  {

    typename PointSetType::Pointer sourceLandMarks;
    typename PointSetType::Pointer targetLandMarks;

    // MITK Point Set?
    
    sourceLandMarks = 
      ReadMITKPointSet<PointType, PointSetType>( args.fileSourceLandmarks, args.flgVerbose );
    
    targetLandMarks = 
      ReadMITKPointSet<PointType, PointSetType>( args.fileTargetLandmarks, args.flgVerbose );
    
    
    // CSV File?
    
    if ( ! sourceLandMarks )
    {
      sourceLandMarks = 
        ReadCSVPointSet<PointType, PointSetType>( args.fileSourceLandmarks, args.flgVerbose );
    }
    
    if ( ! targetLandMarks )
    {
      targetLandMarks = 
        ReadCSVPointSet<PointType, PointSetType>( args.fileTargetLandmarks, args.flgVerbose );
    }



    // Compute the elastic body spline transformation
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    ebs = ElasticBodyTransformType::New();

    // The landmarks have to be swapped to transform the source image into the target space

    ebs->SetSourceLandmarks(targetLandMarks);
    ebs->SetTargetLandmarks(sourceLandMarks);

    ebs->SetAlpha( 12.*( 1. - args.poisson) - 1. );
    ebs->SetStiffness( args.stiffness );

    try
    {
      std::cout << "Computing the elastic body spline matrix" << std::endl;
      ebs->ComputeWMatrix();
    }
    catch( itk::ExceptionObject & err ) 
    { 
      std::cerr << "ERROR: Failed to compute the spline matrix, exception: " << std::endl
                << err << std::endl; 
      return EXIT_FAILURE;
    }


    // Save the matrix to a file?
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~

    if ( args.fileOutputElasticBodySplineMatrix.length() )
    {
      typedef itk::TransformFileWriterTemplate< CoordinateRepType > TransformWriterType;
      typename TransformWriterType::Pointer transformWriter = TransformWriterType::New();
      
      transformWriter->SetInput( ebs );
      
      transformWriter->SetFileName( args.fileOutputElasticBodySplineMatrix );
      
      try
      {
        std::cout << "Writing the elastic body spline matrix to file: " 
                  << args.fileOutputElasticBodySplineMatrix << std::endl;
        transformWriter->Update();       
      }
      catch( itk::ExceptionObject & excp )
      {
        std::cerr << "ERROR: Failed to write the spline matrix, exception: " << std::endl
                  << excp << std::endl;
        return EXIT_FAILURE;
      }
    }

  }


  // Or read the matrix directly?
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  else if ( args.fileInputElasticBodySplineMatrix.length() )
  {
    itk::TransformFactory< ElasticBodyTransformType >::RegisterTransform();

    typedef itk::TransformFileReaderTemplate< CoordinateRepType > TransformReaderType;
    typename TransformReaderType::Pointer transformReader = TransformReaderType::New();
      
    transformReader->SetFileName( args.fileInputElasticBodySplineMatrix );
      
    try
    {
      std::cout << "Reading the elastic body spline matrix from file: " 
                << args.fileInputElasticBodySplineMatrix << std::endl;
      transformReader->Update();       
  
      typedef TransformReaderType::TransformListType *TransformListType;
      TransformListType transforms = transformReader->GetTransformList();
      
      typename itk::TransformFileReader::TransformListType::const_iterator it = transforms->begin();

      ebs = static_cast<ElasticBodyTransformType*>((*it).GetPointer());

      ebs->SetAlpha( 12.*( 1. - args.poisson) - 1. );

      ebs->ComputeWMatrix();
    }
    catch( itk::ExceptionObject & excp )
    {
      std::cerr << "ERROR: Failed to read the spline matrix, exception: " << std::endl
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
  resampler->SetTransform( ebs );

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

    typename ElasticBodyTransformType::InputPointType  point1;
    typename ElasticBodyTransformType::OutputPointType point2;
    typename DisplacementFieldType::IndexType index;

    FieldVectorType displacement;

    while( ! fi.IsAtEnd() )
    {
      index = fi.GetIndex();

      field->TransformIndexToPhysicalPoint( index, point1 );

      point2 = ebs->TransformPoint( point1 );

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
 * \brief Transforms an image using a elastic body spline warp computed from a pair of point sets.
 */

int main(int argc, char** argv)
{
  unsigned int i;

  int result;

  struct arguments args;

  PARSE_ARGS;

  args.flgVerbose = flgVerbose;
  args.flgDebug   = flgDebug;

  args.poisson = poisson;
  args.stiffness = stiffness;

  args.fileOutputImage                 = fileOutputImage;
  args.fileOutputDeformationField      = fileOutputDeformationField;
  args.fileOutputElasticBodySplineMatrix = fileOutputElasticBodySplineMatrix;

  args.fileSourceLandmarks = fileSourceLandmarks;
  args.fileTargetLandmarks = fileTargetLandmarks;
  
  args.fileInputSourceImage = fileInputSourceImage;
  args.fileInputTargetImage = fileInputTargetImage;

  args.fileInputElasticBodySplineMatrix = fileInputElasticBodySplineMatrix;


  if ( ! ( args.fileSourceLandmarks.length() || 
           args.fileTargetLandmarks.length() || 
           args.fileInputElasticBodySplineMatrix.length() ) )
  {
    std::cerr << "ERROR: Two point sets or an input matrix must be specified" << std::endl;
    return EXIT_FAILURE;
  }

  if ( ! args.fileInputSourceImage.length() )
  {
    std::cerr << "ERROR: An input image must be specified" << std::endl;
    return EXIT_FAILURE;
  }

  int dims = itk::PeekAtImageDimensionFromSizeInVoxels(args.fileInputSourceImage);
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
   

  switch (itk::PeekAtComponentType(args.fileInputSourceImage))
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
