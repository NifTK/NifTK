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
#include <itkThinPlateSplineScatteredDataPointSetToImageFilter.h>

#include <niftkThinPlateSplineScatteredDataPointSetToImageCLP.h>

/*!
 * \file niftkThinPlateSplineScatteredDataPointSetToImage.cxx
 * \page niftkThinPlateSplineScatteredDataPointSetToImage
 * \section niftkThinPlateSplineScatteredDataPointSetToImageSummary Computes a thin plate spline mask approximation to a set of landmarks.
 */



struct arguments
{
  bool flgVerbose;
  bool flgDebug;
  bool flgInvert;

  int size[3];

  float resolution[3];
  float origin[3];

  double stiffness;

  std::string dimension;

  std::string fileOutput;

  std::string fileInputMITK;
  std::string fileInputPointSet;

  std::string fileInputImage;

  arguments() {
    flgVerbose = false;
    flgDebug = false;
    flgInvert = false;

    size[0] = 100;
    size[1] = 100;
    size[2] = 100;

    resolution[0] = 1.;
    resolution[1] = 1.;
    resolution[2] = 1.;

    origin[0] = 0.;
    origin[1] = 0.;
    origin[2] = 0.;

    stiffness = 1.;
  }
};


template <int OutputDimension, class PixelType>
int DoMain(arguments args)
{
  typedef itk::Image< PixelType, OutputDimension >                 OutputImageType; 

  typedef itk::ImageFileWriter< OutputImageType >                  OutputImageWriterType;

  typename OutputImageType::RegionType  region;
  typename OutputImageType::IndexType   index;
  typename OutputImageType::SizeType    size;
  typename OutputImageType::SpacingType spacing;
  typename OutputImageType::PointType   origin;

  typedef itk::PointSet<double, OutputDimension> LandmarkPointSetType;

  typedef typename itk::ThinPlateSplineScatteredDataPointSetToImageFilter< LandmarkPointSetType, OutputImageType > ThinPlateSplineFilterType;

  typename OutputImageType::Pointer imMask;


  // Read the input points
  // ~~~~~~~~~~~~~~~~~~~~~

  unsigned int iDim;
  unsigned int iLandmark = 0;
  unsigned int nLandmarks;

  std::fstream fin;

  typedef typename ThinPlateSplineFilterType::LandmarkPointType LandmarkPointType;

  typename LandmarkPointSetType::Pointer PointSet = LandmarkPointSetType::New();

  LandmarkPointType point; 

  // Read the landmark text file

  if ( args.fileInputPointSet.length() )
  {

    if ( args.flgVerbose ) {
      std::cout << "Opening landmarks file: " << args.fileInputPointSet << "..." << std::endl;
      std::cout.flush();
    }

    fin.open( args.fileInputPointSet.c_str(), std::ios::in );

    if ( fin.bad() || fin.fail() ) {
      std::cerr << "ERROR: Failed to open file: " << args.fileInputPointSet << std::endl;
      return EXIT_FAILURE;
    }

    iLandmark = 0;

    if ( args.flgVerbose )
    {
      std::cout << "Landmarks:" << std::endl;
    }

    while ( fin.eof() == 0 )
    {

      for ( iDim=0; iDim<OutputDimension; iDim++ )
      {
        fin >> point[iDim];
        if ( fin.eof() ) break;
      }

      if ( fin.eof() ) break;

      if ( args.flgVerbose )
      {
        std::cout << std::setw(8) << iLandmark << " "
                  << point << std::endl;
      }

      PointSet->SetPoint( iLandmark, point );

      iLandmark++;
    }

    fin.close();
  }


  // Read an MITK point set

  if ( args.fileInputMITK.length() )
  {
    typename itk::DOMNodeXMLReader::Pointer domReader = itk::DOMNodeXMLReader::New();

    domReader->SetFileName( args.fileInputMITK );

    try {
      std::cout << "Reading MITK PointSet file: " << args.fileInputMITK << std::endl;
      domReader->Update();
    }
    catch( itk::ExceptionObject & err )
    {
      std::cerr << "ERROR: Failed to read MITK PointSet file: " << err << std::endl;
      exit( EXIT_FAILURE );
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

          if ( args.flgVerbose )
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

  
  // Fit a thin plate spline to the points
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typename ThinPlateSplineFilterType::Pointer filter = ThinPlateSplineFilterType::New();

  filter->SetInput( PointSet );

  for ( iDim=0; iDim<OutputDimension; iDim++ )
  {
    index[iDim] = 0;
    size[iDim] = args.size[ iDim ];
    spacing[iDim] = args.resolution[ iDim ];
    origin[iDim] = args.origin[ iDim ];
  }

  region.SetSize( size );
  region.SetIndex( index );

  filter->SetSpacing( spacing );
  filter->SetOrigin(  origin );
  filter->SetSize( size );

  if ( args.flgInvert )
  {
    filter->SetInvert( true );
  }

  if ( args.dimension == std::string( "x" ) )
  {
    filter->SetSplineHeightDimension( 0 );
  }
  else if ( ( OutputDimension == 2 ) || ( args.dimension == std::string( "y" ) ) )
  {
    filter->SetSplineHeightDimension( 1 );
  }
  else if ( ( OutputDimension == 3 ) || ( args.dimension == std::string( "z" ) ) )
  {
    filter->SetSplineHeightDimension( 2 );
  }


  filter->SetStiffness( args.stiffness );

  if ( args.flgDebug )
  {
    filter->SetDebug(true);
  }

  try
  {  
    std::cout << "Computing thin plate spline mask" << std::endl;
    filter->Update();
  }                                                                                
  catch (itk::ExceptionObject &e)                                                  
  {                                                                                
    std::cerr << "ERROR: Failed to compute the thin plate spline mask" << std::endl;
    std::cerr << e << std::endl;                                                   
    return EXIT_FAILURE;                                                           
  }                                                                                

  imMask = filter->GetOutput();
  imMask->DisconnectPipeline();


  // Write the image out
  // ~~~~~~~~~~~~~~~~~~~

  if ( args.fileOutput.length() != 0 ) 
  {                                        
    typename OutputImageWriterType::Pointer writer = OutputImageWriterType::New();

    writer->SetFileName( args.fileOutput );
    writer->SetInput( imMask );               

    try
    {  
      std::cout << "Writing the thin plate spline mask to image: " << args.fileOutput << std::endl;
      writer->Update();                                                              
    }                                                                                
    catch (itk::ExceptionObject &e)                                                  
    {                                                                                
      std::cerr << "ERROR: Failed to write the mask to a file" << std::endl;
      std::cerr << e << std::endl;                                                   
      return EXIT_FAILURE;                                                           
    }                                                                                
  }                                                                                  
   


  return EXIT_SUCCESS;
}

/**
 * \brief Determines the input image dimension and pixel type.
 */
int main(int argc, char** argv)
{
  unsigned int i;

  int result;
  int numberOfDimensions;

  struct arguments args;

  PARSE_ARGS;

  args.flgVerbose = flgVerbose;
  args.flgDebug = flgDebug;
  args.flgInvert = flgInvert;

  for ( i=0; (i<size.size()) && (i<3); i++ )
  {
    args.size[i] = size[i];
  }

  for ( i=0; (i<resolution.size()) && (i<3); i++ )
  {
    args.resolution[i] = resolution[i];
  }

  for ( i=0; (i<origin.size()) && (i<3); i++ )
  {
    args.origin[i] = origin[i];
  }
  
  args.dimension = dimension;

  args.stiffness = stiffness;

  args.fileOutput = fileOutputImage;
  args.fileInputMITK = fileInputMITK;
  args.fileInputPointSet = fileInputPointSet;
  
  args.fileInputImage = fileInputImage;


  if ( args.size[2] > 1 )
  {
    numberOfDimensions = 3;
  }
  else if ( args.size[1] > 1 )
  {
    numberOfDimensions = 2;
  }
  else if ( args.size[0] > 1 )
  {
    numberOfDimensions = 1;
  }
  else
  {
    numberOfDimensions = 0;
  }


  if ( ! ( args.fileInputMITK.length() || args.fileInputPointSet.length() ) )
  {
    std::cerr << "ERROR: One or more point sets must be specified" << std::endl;
    return EXIT_FAILURE;
  }

  if ( args.fileInputImage.length() > 0 )
  {
    itk::ImageIOBase::Pointer imageIO;
    itk::InitialiseImageIO( args.fileInputImage, imageIO );

    numberOfDimensions = imageIO->GetNumberOfDimensions();

    if ( numberOfDimensions > 0 )
    {
      args.size[0] = imageIO->GetDimensions(0);
      args.origin[0] = imageIO->GetOrigin(0);
      args.resolution[0] = imageIO->GetSpacing(0);
    }

    if ( numberOfDimensions > 1 )
    {
      args.size[1] = imageIO->GetDimensions(1);
      args.origin[1] = imageIO->GetOrigin(1);
      args.resolution[1] = imageIO->GetSpacing(1);
    }

    if ( numberOfDimensions > 2 )
    {
      args.size[2] = imageIO->GetDimensions(2);
      args.origin[2] = imageIO->GetOrigin(2);
      args.resolution[2] = imageIO->GetSpacing(2);
    }
  }

  switch ( numberOfDimensions )
  {

  case 3:
  {
    result = DoMain<3, unsigned char>( args );  
    break;
  }

  case 2:
  {
    result = DoMain<2, unsigned char>( args );  
    break;
  }

  default:
  {
    std::cerr << "ERROR: Unsupported number of dimensions (" << numberOfDimensions << ")"
              << std::endl;
    result = EXIT_FAILURE;
  }
  }

  return result;
}
