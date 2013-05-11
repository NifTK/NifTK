/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

/*!
 * \file niftkCreateMaskImage.cxx
 * \page niftkCreateMaskImage
 * \section niftkCreateMaskImageSummary niftkCreateMaskImage creates a mask image using the input target image and a voxel-wise bounding box.
 */

#include <itkLogHelper.h>
#include <itkCommandLineHelper.h>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkNumericTraits.h>
#include <itkImageDuplicator.h>
#include <itkImageRegionIteratorWithIndex.h>

#include <niftkCreateMaskImageCLP.h>


//  -------------------------------------------------------------------------
//  arguments
//  -------------------------------------------------------------------------

struct arguments
{
  bool flgIndicesInMM;

  float sx;
  float sy;
  float sz;
  float st;

  float ex;
  float ey;
  float ez;
  float et;

  std::string fileInputImage;
  std::string fileOutputImage;

  arguments() {

    flgIndicesInMM = false;

    sx = 0.;
    sy = 0.;
    sz = 0.;
    st = 0.;
    
    ex = 0.;
    ey = 0.;
    ez = 0.;
    et = 0.;
  }
};



//  -------------------------------------------------------------------------
//  CreateMaskImage()
/// \brief Creates a mask using the ROI specified by the input image.
//  -------------------------------------------------------------------------

template <int Dimension, class PixelType>
int CreateMaskImage( arguments &args )
{

  typedef itk::Image< PixelType, Dimension > ImageType;   
  typedef itk::ImageFileReader< ImageType >  InputImageReaderType;
  typedef itk::ImageFileWriter< ImageType >  OutputImageWriterType;

  typedef itk::ImageRegionIteratorWithIndex< ImageType > IteratorType;    

  typename ImageType::Pointer maskImage = 0;

  float startCoord[4];
  float endCoord[4];


  startCoord[0] = args.sx;
  startCoord[1] = args.sy;
  startCoord[2] = args.sz;
  startCoord[3] = args.st;

  endCoord[0] = args.ex;
  endCoord[1] = args.ey;
  endCoord[2] = args.ez;
  endCoord[3] = args.et;


  // Read the input image
  // ~~~~~~~~~~~~~~~~~~~~

  typename InputImageReaderType::Pointer imageReader = InputImageReaderType::New();
  
  imageReader->SetFileName( args.fileInputImage );
  
  try
  {
    std::cout << "Reading input image: " << args.fileInputImage << std::endl; 
    imageReader->Update(); 
  }
  catch( itk::ExceptionObject & err ) 
  { 
    std::cerr << std::endl << "ERROR: Failed to read the input image: " << err
	      << std::endl << std::endl; 
    return EXIT_FAILURE;
  }                
  
  maskImage = imageReader->GetOutput();
  maskImage->DisconnectPipeline();

  maskImage->FillBuffer( static_cast<PixelType>(0) );


  // Calculate the extent of the bounding specified (the whole image by default)
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  unsigned int i;
  typename ImageType::IndexType idxStart, idxEnd;
  typename ImageType::RegionType region = maskImage->GetLargestPossibleRegion();
  typename ImageType::SizeType size = region.GetSize();
  
  // Convert the indices from mm?

  if ( args.flgIndicesInMM ) 
  {
    typename ImageType::PointType ptStart, ptEnd;

    for ( i=0; i<Dimension; i++ )
    {
      ptStart[i] = startCoord[i];
      ptEnd[i]   =   endCoord[i];
    }

    maskImage->TransformPhysicalPointToIndex( ptStart, idxStart );
    maskImage->TransformPhysicalPointToIndex(   ptEnd,   idxEnd );
  }
  else
  {
    for ( i=0; i<Dimension; i++ )
    { 
      if ( startCoord[i] < 0 )
	idxStart[i] = 0;
      else if ( startCoord[i] > size[i] - 1 )
	idxStart[i] = size[i] - 1;
      else
	idxStart[i] = static_cast<int>( startCoord[i] );
      
      if ( endCoord[i] < 0 )
	idxEnd[i] = 0;
      else if ( endCoord[i] > size[i] - 1 )
	idxEnd[i] = size[i] - 1;
      else
	idxEnd[i]   = static_cast<int>( endCoord[i] );
    }
  }

  // Clip the indices to the image size
  
  unsigned int idxTmp;

  for ( i=0; i<Dimension; i++ )
  { 
    if ( idxStart[i] < 0 ) idxStart[i] = 0;
    if ( idxStart[i] > static_cast<int>( size[i] - 1 ) ) idxStart[i] = size[i] - 1;
  
    if ( idxEnd[i] < 0 ) idxEnd[i] = 0;
    if ( idxEnd[i] > static_cast<int>( size[i] - 1 ) ) idxEnd[i] = size[i] - 1;
  
    if ( idxStart[i] > idxEnd[i] ) {
      idxTmp = idxEnd[i];
      idxEnd[i] = idxStart[i];
      idxStart[i] = idxTmp;
    }

    size[i] = idxEnd[i] - idxStart[i] + 1;
  }

  std::cout << "Mask bounding box:" << std::endl
	    << "   from: " << idxStart << std::endl
	    << "   to:   " << idxEnd << std::endl
	    << "   size: " << size << std::endl;
  

  // Set voxels inside the bounding box
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
  region.SetSize( size );
  region.SetIndex( idxStart );

  IteratorType inputIterator( maskImage, region );
  
  for ( inputIterator.GoToBegin(); 
	! inputIterator.IsAtEnd();
	++inputIterator )
  {
    inputIterator.Set( 1 );
  }


  // Write the mask image to a file
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    

  typename OutputImageWriterType::Pointer imageWriter = OutputImageWriterType::New();

  imageWriter->SetFileName( args.fileOutputImage );
  imageWriter->SetInput( maskImage );
  
  try
  {
    std::cout << "Writing cropped image to: " << args.fileOutputImage << std::endl; 
    imageWriter->Update(); 
  }
  catch( itk::ExceptionObject & err ) 
  { 
    std::cerr << std::endl << "ERROR: Failed to crop and write the image to a file." << err
	      << std::endl << std::endl; 
    return EXIT_FAILURE;
  }                

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

  if ( fileInputImage.length() == 0 || fileOutputImage.length() == 0 )
  {
    commandLine.getOutput()->usage(commandLine);
    return EXIT_FAILURE;
  }

  args.flgIndicesInMM = flgIndicesInMM;

  args.sx = sx;
  args.sy = sy;
  args.sz = sz;
  args.st = st;

  args.ex = ex;
  args.ey = ey;
  args.ez = ez;
  args.et = et;

  args.fileInputImage  = fileInputImage;
  args.fileOutputImage = fileOutputImage;
  

  // Create a mask of the correct dimension

  int result = 0;
  int dims = itk::PeekAtImageDimensionFromSizeInVoxels( args.fileInputImage );

  switch ( dims )
  {
  case 2:
    result = CreateMaskImage<2, unsigned char>( args );
    break;

  case 3:
    result = CreateMaskImage<3, unsigned char>( args );
    break;

  case 4:
    result = CreateMaskImage<4, unsigned char>( args );
    break;

  default:
    std::cerr << "ERROR: Unsupported image dimension: " << dims << std::endl;
    return EXIT_FAILURE;
  }

  return result;  
}
