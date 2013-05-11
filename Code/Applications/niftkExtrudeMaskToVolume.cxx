/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

/*!
 * \file niftkExtrudeMaskToVolume.cxx
 * \page niftkExtrudeMaskToVolume
 * \section niftkExtrudeMaskToVolumeSummary Converts an image
 * containing one or more masks in the 'x', 'y' or 'z' planes into a
 * volume by extruding these masks across the whole volume. This
 * program was created to enable simple orthogonal planar masks (such as those
 * that can be created with the MITK Segmentation Plugin in NiftyView) 
 * which have been drawn on a single slice or slices,
 * to be converted into a crude volume mask. Such a mask could then be
 * used to mask the target or moving image in a registration, for instance.
 *
 */


#include <itkLogHelper.h>
#include <itkCommandLineHelper.h>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImageRegionIteratorWithIndex.h>

#include <niftkExtrudeMaskToVolumeCLP.h>

#include <boost/date_time/posix_time/posix_time.hpp>

#ifndef HZ
  #if defined(__APPLE__)
    #define HZ __DARWIN_CLK_TCK
  #endif
#endif


//  -------------------------------------------------------------------------
//  arguments
//  -------------------------------------------------------------------------

struct arguments
{
  std::string fileInputMask;
  std::string fileOutputMask;

  bool flgExtrudeInX;
  bool flgExtrudeInY;
  bool flgExtrudeInZ;

  arguments() {

    flgExtrudeInX = true;
    flgExtrudeInY = true;
    flgExtrudeInZ = true;
  }
};


// --------------------------------------------------------------------------
// WriteImageToFile()
// --------------------------------------------------------------------------

template <class TImageType>
bool WriteImageToFile( std::string fileOutput, typename TImageType::Pointer image )
{
  if ( fileOutput.length() ) {

    typedef itk::ImageFileWriter< TImageType > FileWriterType;

    typename FileWriterType::Pointer writer = FileWriterType::New();

    writer->SetFileName( fileOutput.c_str() );
    writer->SetInput( image );

    try
    {
      std::cout << "Writing image to file: "
		<< fileOutput.c_str() << std::endl;
      writer->Update();
    }
    catch (itk::ExceptionObject &e)
    {
      std::cerr << e << std::endl;
    }

    return true;
  }
  else
    return false;
}


//  -------------------------------------------------------------------------
//  DoMain()
/// \brief Takes a volume containing planar masks and extrudes these across the whole volume.
//  -------------------------------------------------------------------------

template <int Dimension, class PixelType>
int DoMain( arguments &args )
{
  float progress = 0.;

  typedef itk::Image< PixelType, Dimension > ImageType;   

  typedef itk::ImageFileReader< ImageType >  InputImageReaderType;
  typedef itk::ImageFileWriter< ImageType >  OutputImageWriterType;

  typedef itk::ImageRegionIteratorWithIndex< ImageType > IteratorType;
  
  typename ImageType::Pointer image = 0;

  
  std::cout << "ExtrudeInX: " << args.flgExtrudeInX << std::endl
	    << "ExtrudeInY: " << args.flgExtrudeInY << std::endl
	    << "ExtrudeInZ: " << args.flgExtrudeInZ << std::endl;


  // Initialise the start time
  // ~~~~~~~~~~~~~~~~~~~~~~~~~

  boost::posix_time::ptime startTime = boost::posix_time::second_clock::local_time();


  // Output the slicer execution model XML
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  std::cout << "<filter-start>" << std::endl

	    << " <filter-name>" << std::endl
	    << "niftkExtrudeMaskToVolume" << std::endl
	    << " </filter-name>" << std::endl

	    << " <filter-comment>" << std::endl
	    << "Takes a volume containing planar masks and " 
	    << "extrudes these across the whole volume." << std::endl
	    << " </filter-comment>" << std::endl

	    << "</filter-start>" << std::endl;


  std::cout << "<filter-progress>" << std::endl
	    << progress << std::endl
	    << "</filter-progress>" << std::endl;


  // Read the input image
  // ~~~~~~~~~~~~~~~~~~~~

  typename InputImageReaderType::Pointer reader = InputImageReaderType::New();

  reader->SetFileName( args.fileInputMask.c_str() );

  try
  { 
    std::cout << "Reading image: " << args.fileInputMask << std::endl;
    reader->Update();
  }
  catch (itk::ExceptionObject &ex)
  { 
    std::cerr << "ERROR: reading image: " << args.fileInputMask.c_str()
	       << std::endl << ex << std::endl;
    return EXIT_FAILURE;
  }

  image = reader->GetOutput();
  image->DisconnectPipeline();

  
  progress = 0.2;
  std::cout << "<filter-progress>" << std::endl
	    << progress << std::endl
	    << "</filter-progress>" << std::endl;


  // Create the 2D masks at each orientation
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
  // Get the volume size

  typename ImageType::RegionType  regionVolume;
  typename ImageType::IndexType   startVolume;
  typename ImageType::SizeType    sizeVolume;
  typename ImageType::SpacingType spacingVolume;
  typename ImageType::PointType   originVolume;

  regionVolume = image->GetLargestPossibleRegion();

  sizeVolume  = regionVolume.GetSize();
  startVolume = regionVolume.GetIndex();

  spacingVolume = image->GetSpacing();
  originVolume  = image->GetOrigin();

  std::cout << "Dimensions: " 
	    << sizeVolume[0] << ", " 
	    << sizeVolume[1] << ", " 
	    << sizeVolume[2] << std::endl
	    << "Start: " 
	    << startVolume[0] << ", " 
	    << startVolume[1] << ", " 
	    << startVolume[2] << std::endl
	    << "Spacing: " 
	    << spacingVolume[0] << ", " 
	    << spacingVolume[1] << ", " 
	    << spacingVolume[2] << std::endl
	    << "Origin: " 
	    << originVolume[0] << ", " 
	    << originVolume[1] << ", " 
	    << originVolume[2] << std::endl;
       

  // Set the plane size

  typedef itk::Image< PixelType, 2 > PlaneType;

  typename PlaneType::RegionType  regionPlane;
  typename PlaneType::IndexType   startPlane;
  typename PlaneType::SizeType    sizePlane;
  typename PlaneType::SpacingType spacingPlane;
  typename PlaneType::PointType   originPlane;

  // 'xy'

  sizePlane[0] = sizeVolume[0];
  sizePlane[1] = sizeVolume[1];

  startPlane[0] = startVolume[0];
  startPlane[1] = startVolume[1];

  spacingPlane[0] = spacingVolume[0];
  spacingPlane[1] = spacingVolume[1];
  
  originPlane[0] = originVolume[0];
  originPlane[1] = originVolume[1];
  
  typename PlaneType::Pointer planeXY = PlaneType::New();

  regionPlane.SetSize( sizePlane );
  regionPlane.SetIndex( startPlane );

  planeXY->SetRegions( regionPlane );
  planeXY->SetSpacing( spacingPlane );
  planeXY->SetOrigin( originPlane );

  planeXY->Allocate();  
  planeXY->FillBuffer( 0 );

  progress = 0.2666;
  std::cout << "<filter-progress>" << std::endl
	    << progress << std::endl
	    << "</filter-progress>" << std::endl;

  // 'yz'

  sizePlane[0] = sizeVolume[1];
  sizePlane[1] = sizeVolume[2];

  startPlane[0] = startVolume[1];
  startPlane[1] = startVolume[2];

  spacingPlane[0] = spacingVolume[1];
  spacingPlane[1] = spacingVolume[2];
  
  originPlane[0] = originVolume[1];
  originPlane[1] = originVolume[2];
  
  typename PlaneType::Pointer planeYZ = PlaneType::New();

  regionPlane.SetSize( sizePlane );
  regionPlane.SetIndex( startPlane );

  planeYZ->SetRegions( regionPlane );
  planeYZ->SetSpacing( spacingPlane );
  planeYZ->SetOrigin( originPlane );

  planeYZ->Allocate();  
  planeYZ->FillBuffer( 0 );

  progress = 0.3333;
  std::cout << "<filter-progress>" << std::endl
	    << progress << std::endl
	    << "</filter-progress>" << std::endl;

  // 'xz'

  sizePlane[0] = sizeVolume[0];
  sizePlane[1] = sizeVolume[2];

  startPlane[0] = startVolume[0];
  startPlane[1] = startVolume[2];

  spacingPlane[0] = spacingVolume[0];
  spacingPlane[1] = spacingVolume[2];
  
  originPlane[0] = originVolume[0];
  originPlane[1] = originVolume[2];
  
  typename PlaneType::Pointer planeXZ = PlaneType::New();

  regionPlane.SetSize( sizePlane );
  regionPlane.SetIndex( startPlane );

  planeXZ->SetRegions( regionPlane );
  planeXZ->SetSpacing( spacingPlane );
  planeXZ->SetOrigin( originPlane );

  planeXZ->Allocate();  
  planeXZ->FillBuffer( 0 );

  progress = 0.4;
  std::cout << "<filter-progress>" << std::endl
	    << progress << std::endl
	    << "</filter-progress>" << std::endl;


  // Iterate through the image setting the 2D masks
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  IteratorType iter( image, regionVolume );

  typename ImageType::IndexType idxVolume;
  typename PlaneType::IndexType idxPlane;

  for ( iter.GoToBegin(); ! iter.IsAtEnd(); ++iter )
  {
    if ( iter.Get() )
    {
      idxVolume = iter.GetIndex();

      // xy

      if ( args.flgExtrudeInZ ) 
      {
      
	idxPlane[0] = idxVolume[0];
	idxPlane[1] = idxVolume[1];
      
	planeXY->SetPixel( idxPlane, 1 );
      }

      // yz

      if ( args.flgExtrudeInX ) 
      {
	idxPlane[0] = idxVolume[1];
	idxPlane[1] = idxVolume[2];
      
	planeYZ->SetPixel( idxPlane, 1 );
      }

      // xz

      if ( args.flgExtrudeInY ) 
      {
	idxPlane[0] = idxVolume[0];
	idxPlane[1] = idxVolume[2];
      
	planeXZ->SetPixel( idxPlane, 1 );
      }
    }
  }

#if 0
  WriteImageToFile< PlaneType >( std::string( "planeXY.nii.gz"), planeXY );
  WriteImageToFile< PlaneType >( std::string( "planeYZ.nii.gz"), planeYZ );
  WriteImageToFile< PlaneType >( std::string( "planeXZ.nii.gz"), planeXZ );
#endif

  progress = 0.6;
  std::cout << "<filter-progress>" << std::endl
	    << progress << std::endl
	    << "</filter-progress>" << std::endl;

  
  // and then extrude the masks back through the volume
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  image->FillBuffer( 0 );

  for ( iter.GoToBegin(); ! iter.IsAtEnd(); ++iter )
  {
    idxVolume = iter.GetIndex();

    // xy

    if ( args.flgExtrudeInZ ) 
    {
      
      idxPlane[0] = idxVolume[0];
      idxPlane[1] = idxVolume[1];
      
      if ( ! planeXY->GetPixel( idxPlane ) )
	continue;
    }

    // yz

    if ( args.flgExtrudeInX ) 
    {

      idxPlane[0] = idxVolume[1];
      idxPlane[1] = idxVolume[2];
      
      if ( ! planeYZ->GetPixel( idxPlane ) )
	continue;
    }

    // xz

    if ( args.flgExtrudeInY ) 
    {

      idxPlane[0] = idxVolume[0];
      idxPlane[1] = idxVolume[2];
      
      if ( ! planeXZ->GetPixel( idxPlane ) )
	continue;
    }

    // All masks are set so include this voxel

    
    iter.Set( 1 );
  }

  progress = 0.8;
  std::cout << "<filter-progress>" << std::endl
	    << progress << std::endl
	    << "</filter-progress>" << std::endl;
  

  // Save the extruded volume to a file
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( ! WriteImageToFile< ImageType >( args.fileOutputMask, image ) )
    return EXIT_FAILURE;

  progress = 1.0;
  std::cout << "<filter-progress>" << std::endl
	    << progress << std::endl
	    << "</filter-progress>" << std::endl;


  // Calculate the execution time
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  boost::posix_time::ptime endTime = boost::posix_time::second_clock::local_time();
  boost::posix_time::time_duration duration = endTime - startTime;

  std::cout << "Execution time: " << boost::posix_time::to_simple_string(duration) << std::endl;


  // Output the slicer execution model XML
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  std::cout << "<filter-end>" << std::endl

	    << " <filter-name>" << std::endl
	    << "niftkExtrudeMaskToVolume" << std::endl
	    << " </filter-name>" << std::endl

	    << " <filter-time>" << std::endl
	    << boost::posix_time::to_simple_string(duration) << std::endl
	    << " </filter-time>" << std::endl

	    << "</filter-end>" << std::endl;

  return EXIT_SUCCESS;  
}


// -------------------------------------------------------------------------
// main()
// -------------------------------------------------------------------------

int main( int argc, char *argv[] )
{
  arguments args;


  // Validate command line args
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~

  PARSE_ARGS;

  if ( fileInputMask.length() == 0 || fileOutputMask.length() == 0 )
  {
    commandLine.getOutput()->usage(commandLine);
    return EXIT_FAILURE;
  }

  args.fileInputMask  = fileInputMask;
  args.fileOutputMask   = fileOutputMask;

  args.flgExtrudeInX = flgExtrudeInX;
  args.flgExtrudeInY = flgExtrudeInY;
  args.flgExtrudeInZ = flgExtrudeInZ;

  if ( ! ( args.flgExtrudeInX || args.flgExtrudeInY || args.flgExtrudeInZ ) )
  {
    std::cerr << "ERROR: One of --xAxis, --yAxis or --zAxis must be specified" << std::endl;
    return EXIT_FAILURE;
  }


  // Find the image dimension and the image type

  int result = 0;
  int dims = itk::PeekAtImageDimensionFromSizeInVoxels( args.fileInputMask );
  
  switch ( dims )
  {

  case 3:
  {
    switch ( itk::PeekAtComponentType( args.fileInputMask ) )
    {
    case itk::ImageIOBase::UCHAR:
      result = DoMain<3, unsigned char>( args );
      break;
      
    case itk::ImageIOBase::CHAR:
      result = DoMain<3, char>( args );
      break;
      
    case itk::ImageIOBase::USHORT:
      result = DoMain<3, unsigned short>( args );
      break;
      
    case itk::ImageIOBase::SHORT:
      result = DoMain<3, short>( args );
      break;
      
    case itk::ImageIOBase::UINT:
      result = DoMain<3, unsigned int>( args );
      break;
      
    case itk::ImageIOBase::INT:
      result = DoMain<3, int>( args );
      break;
      
    case itk::ImageIOBase::ULONG:
      result = DoMain<3, unsigned long>( args );
      break;
      
    case itk::ImageIOBase::LONG:
      result = DoMain<3, long>( args );
      break;
      
    case itk::ImageIOBase::FLOAT:
      result = DoMain<3, float>( args );
      break;
      
    case itk::ImageIOBase::DOUBLE:
      result = DoMain<3, double>( args );
      break;
      
    default:
      std::cerr << "ERROR: Non standard pixel format" << std::endl;
      return EXIT_FAILURE;
    }
    break;
  }

  default:
    std::cerr << "ERROR: Unsupported image dimension: " << dims << std::endl;
    return EXIT_FAILURE;
  }

  return result;  
}

