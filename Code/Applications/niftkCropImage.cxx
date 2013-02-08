/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

/*!
 * \file niftkCrop.cxx
 * \page niftkCrop
 * \section niftkCropSummary Crops the input image using a mask and/or a voxel-wise bounding box.
 *
 * \li Pixel type: Scalars only, of type short.
 *
 * \section niftkCropCaveat Caveats
 * \li Image headers not checked. By "voxel by voxel basis" we mean that the image geometry, origin, orientation is not checked.
 */


#include "itkLogHelper.h"
#include "itkCropImageFilter.h"
#include "itkCommandLineHelper.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkNumericTraits.h"
#include "itkImageDuplicator.h"
#include "itkImageRegionIteratorWithIndex.h"

#include "niftkCropImageCLP.h"


//  -------------------------------------------------------------------------
//  arguments
//  -------------------------------------------------------------------------

struct arguments
{
  bool flgCombineMaskAndBoundingBoxViaUnion;
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
  std::string fileMaskImage;
  std::string fileOutputImage;

  arguments() {

    flgCombineMaskAndBoundingBoxViaUnion = false;
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
//  CropImage()
/// \brief Takes mask and target and crops target where mask is non zero.
//  -------------------------------------------------------------------------

template <int Dimension, class PixelType>
int CropImage( arguments &args )
{

  typedef itk::Image< PixelType, Dimension > ImageType;   
  typedef itk::ImageFileReader< ImageType >  InputImageReaderType;
  typedef itk::ImageFileWriter< ImageType >  OutputImageWriterType;

  typedef itk::CropImageFilter< ImageType, ImageType > CropFilterType;

  typedef itk::ImageRegionIteratorWithIndex< ImageType > IteratorType;    

  typename ImageType::Pointer inImage = 0;
  typename ImageType::Pointer maskImage = 0;

  typename ImageType::IndexType idx;

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
  
  inImage = imageReader->GetOutput();
  inImage->DisconnectPipeline();


  // Calculate the extent of the bounding specified (the whole image by default)
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  unsigned int i;
  typename ImageType::IndexType idxStart, idxEnd;
  typename ImageType::RegionType region = inImage->GetLargestPossibleRegion();
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

    inImage->TransformPhysicalPointToIndex( ptStart, idxStart );
    inImage->TransformPhysicalPointToIndex(   ptEnd,   idxEnd );
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

  std::cout << "Crop bounding box:" << std::endl
	    << "   from: " << idxStart << std::endl
	    << "   to:   " << idxEnd << std::endl
	    << "   size: " << size << std::endl;
  

  // Read the input mask and calculate the combined bounding box
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( args.fileMaskImage.length() )
  {
    typename InputImageReaderType::Pointer maskReader = InputImageReaderType::New();

    maskReader->SetFileName( args.fileMaskImage );
  
    try
    {
      std::cout << "Reading mask image: " << args.fileMaskImage << std::endl; 
      maskReader->Update(); 
    }
    catch( itk::ExceptionObject & err ) 
    { 
      std::cerr << std::endl << "ERROR: Failed to read the mask image: " << err
		<< std::endl << std::endl; 
      return EXIT_FAILURE;
    }                

    maskImage = maskReader->GetOutput();
    maskImage->DisconnectPipeline();

    typename ImageType::RegionType maskRegion = maskImage->GetLargestPossibleRegion();
    
    if (  maskRegion != region )
    {
      std::cerr << std::endl 
		<< "ERROR: Input image and mask must have the same dimensions."
		<< std::endl 
		<< "Input region: " << region
		<< "Mask  region: " << maskRegion
		<< std::endl;
      return EXIT_FAILURE;
    }
    
    if ( maskImage->GetSpacing() != inImage->GetSpacing() )
    {
      std::cerr << std::endl 
		<< "ERROR: Input image and mask must have the same resolution."
		<< std::endl 
		<< "Input resolution: " << inImage->GetSpacing() << std::endl 
		<< "Mask  resolution: " << maskImage->GetSpacing() << std::endl 
		<< std::endl;
      return EXIT_FAILURE;
    }

    // The mask and the bounding box combine...?

    if ( args.flgCombineMaskAndBoundingBoxViaUnion )
    {
      typename ImageType::RegionType region = inImage->GetLargestPossibleRegion();
      
      IteratorType inputIterator( maskImage, region );
      
      for ( inputIterator.GoToBegin(); 
	    ! inputIterator.IsAtEnd();
	    ++inputIterator )
      {
	if ( inputIterator.Get() ) {
	  idx = inputIterator.GetIndex();
	  
	  for ( i=0; i<Dimension; i++ )
	  {
	    if ( idx[i] < idxStart[i] ) 
	      idxStart[i] = idx[i];
	      
	    if ( idx[i] > idxEnd[i] )
	      idxEnd[i] = idx[i];
	  }
	}
      }
    }
  

    // ...or only where the mask and bounding box overlap?
    
    else
    {
      typename ImageType::RegionType region = inImage->GetLargestPossibleRegion();
      
      region.SetSize( size );
      region.SetIndex( idxStart );

      idx = idxStart;
      idxStart = idxEnd;
      idxEnd = idx;

      IteratorType inputIterator( maskImage, region );
      
      for ( inputIterator.GoToBegin(); 
	    ! inputIterator.IsAtEnd();
	    ++inputIterator )
      {
	if ( inputIterator.Get( ) )
	{
	  idx = inputIterator.GetIndex();
	
	  for ( i=0; i<Dimension; i++ )
	  {
	    if ( idx[i] < idxStart[i] ) 
	      idxStart[i] = idx[i];
	      
	    if ( idx[i] > idxEnd[i] )
	      idxEnd[i] = idx[i];
	  }
	}
      }
    }
  }



  // Crop the image

  typename CropFilterType::Pointer filter = CropFilterType::New();  

  typename ImageType::SizeType lowerSize, upperSize;
  
  for ( i=0; i<Dimension; i++ )
    lowerSize[i] = idxStart[i];

  filter->SetLowerBoundaryCropSize( lowerSize );


  for ( i=0; i<Dimension; i++ )
    upperSize[i] = inImage->GetLargestPossibleRegion().GetSize()[i] - idxEnd[i];

  filter->SetUpperBoundaryCropSize( upperSize );


  filter->SetInput( inImage );



  // Write the cropped image to a file

  typename OutputImageWriterType::Pointer imageWriter = OutputImageWriterType::New();

  imageWriter->SetFileName( args.fileOutputImage );
  imageWriter->SetInput( filter->GetOutput() );
  
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

  args.flgCombineMaskAndBoundingBoxViaUnion = flgCombineMaskAndBoundingBoxViaUnion;
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
  args.fileMaskImage   = fileMaskImage;
  args.fileOutputImage = fileOutputImage;
  


  // Find the image dimension and the image type

  int result = 0;
  int dims = itk::PeekAtImageDimensionFromSizeInVoxels( args.fileInputImage );
  
  switch ( dims )
  {
  case 2: 
  {
    switch ( itk::PeekAtComponentType( args.fileInputImage ) )
    {
    case itk::ImageIOBase::UCHAR:
      result = CropImage<2, unsigned char>( args );
      break;
      
    case itk::ImageIOBase::CHAR:
      result = CropImage<2, char>( args );
      break;
      
    case itk::ImageIOBase::USHORT:
      result = CropImage<2, unsigned short>( args );
      break;
      
    case itk::ImageIOBase::SHORT:
      result = CropImage<2, short>( args );
      break;
      
    case itk::ImageIOBase::UINT:
      result = CropImage<2, unsigned int>( args );
      break;
      
    case itk::ImageIOBase::INT:
      result = CropImage<2, int>( args );
      break;
      
    case itk::ImageIOBase::ULONG:
      result = CropImage<2, unsigned long>( args );
      break;
      
    case itk::ImageIOBase::LONG:
      result = CropImage<2, long>( args );
      break;
      
    case itk::ImageIOBase::FLOAT:
      result = CropImage<2, float>( args );
      break;
      
    case itk::ImageIOBase::DOUBLE:
      result = CropImage<2, double>( args );
      break;
      
    default:
      std::cerr << "ERROR: Non standard pixel format" << std::endl;
      return EXIT_FAILURE;
    }
    break;
  }

  case 3:
  {
    switch ( itk::PeekAtComponentType( args.fileInputImage ) )
    {
    case itk::ImageIOBase::UCHAR:
      result = CropImage<3, unsigned char>( args );
      break;
      
    case itk::ImageIOBase::CHAR:
      result = CropImage<3, char>( args );
      break;
      
    case itk::ImageIOBase::USHORT:
      result = CropImage<3, unsigned short>( args );
      break;
      
    case itk::ImageIOBase::SHORT:
      result = CropImage<3, short>( args );
      break;
      
    case itk::ImageIOBase::UINT:
      result = CropImage<3, unsigned int>( args );
      break;
      
    case itk::ImageIOBase::INT:
      result = CropImage<3, int>( args );
      break;
      
    case itk::ImageIOBase::ULONG:
      result = CropImage<3, unsigned long>( args );
      break;
      
    case itk::ImageIOBase::LONG:
      result = CropImage<3, long>( args );
      break;
      
    case itk::ImageIOBase::FLOAT:
      result = CropImage<3, float>( args );
      break;
      
    case itk::ImageIOBase::DOUBLE:
      result = CropImage<3, double>( args );
      break;
      
    default:
      std::cerr << "ERROR: Non standard pixel format" << std::endl;
      return EXIT_FAILURE;
    }
    break;
  }

  case 4:
  {
    switch ( itk::PeekAtComponentType( args.fileInputImage ) )
    {
    case itk::ImageIOBase::UCHAR:
      result = CropImage<4, unsigned char>( args );
      break;
      
    case itk::ImageIOBase::CHAR:
      result = CropImage<4, char>( args );
      break;
      
    case itk::ImageIOBase::USHORT:
      result = CropImage<4, unsigned short>( args );
      break;
      
    case itk::ImageIOBase::SHORT:
      result = CropImage<4, short>( args );
      break;
      
    case itk::ImageIOBase::UINT:
      result = CropImage<4, unsigned int>( args );
      break;
      
    case itk::ImageIOBase::INT:
      result = CropImage<4, int>( args );
      break;
      
    case itk::ImageIOBase::ULONG:
      result = CropImage<4, unsigned long>( args );
      break;
      
    case itk::ImageIOBase::LONG:
      result = CropImage<4, long>( args );
      break;
      
    case itk::ImageIOBase::FLOAT:
      result = CropImage<4, float>( args );
      break;
      
    case itk::ImageIOBase::DOUBLE:
      result = CropImage<4, double>( args );
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
