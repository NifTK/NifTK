/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

/*!
 * \file niftkApplyMaskToImage.cxx
 * \page niftkApplyMaskToImage
 * \section niftkApplyMaskToImageSummary Masks the input image using a mask and/or a voxel-wise bounding box.
 *
 * \li Pixel type: Scalars only, of type short.
 *
 * \section niftkApplyMaskToImageCaveat Caveats
 * \li Image headers not checked. By "voxel by voxel basis" we mean that the image geometry, origin, orientation is not checked.
 */

#include "itkLogHelper.h"
#include "itkCommandLineHelper.h"
#include "itkCropTargetImageWhereSourceImageNonZero.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkNumericTraits.h"
#include "itkImageDuplicator.h"

#include "niftkApplyMaskToImageCLP.h"

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
//  MaskImage()
/// \brief Sets the target image to zero where mask is zero and/or inside bounding box.
//  -------------------------------------------------------------------------

template <int Dimension, class PixelType>
int MaskImage( arguments &args )
{

  typedef itk::Image< PixelType, Dimension > ImageType;   
  typedef itk::ImageFileReader< ImageType >  InputImageReaderType;
  typedef itk::ImageFileWriter< ImageType >  OutputImageWriterType;

  typedef itk::CropTargetImageWhereSourceImageNonZeroImageFilter< ImageType, ImageType > MaskFilterType;

  typedef itk::ImageRegionIteratorWithIndex< ImageType > IteratorType;    

  typename ImageType::Pointer inImage = 0;
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
    std::cerr << "ERROR: Failed to read the input image: " << err << std::endl; 
    return EXIT_FAILURE;
  }                
  
  inImage = imageReader->GetOutput();
  inImage->DisconnectPipeline();
  
  typename ImageType::RegionType region = inImage->GetLargestPossibleRegion();
  typename ImageType::SizeType size = region.GetSize();


  // Read the input mask...
  // ~~~~~~~~~~~~~~~~~~~~~~

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
      std::cerr << "ERROR: Failed to read the mask image: " << err << std::endl; 
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
  }

  // ...or create one

  else
  {      
    typedef itk::ImageDuplicator< ImageType > DuplicatorType;

    typename DuplicatorType::Pointer duplicator = DuplicatorType::New();

    duplicator->SetInputImage( inImage );
    duplicator->Update();

    maskImage = duplicator->GetOutput();
    maskImage->DisconnectPipeline();

    if ( args.flgCombineMaskAndBoundingBoxViaUnion )
      maskImage->FillBuffer( 0 );
    else
      maskImage->FillBuffer( 1 );
  }


  // Add a bounding box to the image
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  unsigned int i;
  typename ImageType::IndexType idxStart, idxEnd;
  
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

  std::cout << "Mask bounding box:" << std::endl
	    << "   from: " << idxStart << std::endl
	    << "   to:   " << idxEnd << std::endl
	    << "   size: " << size << std::endl;


  // Set the voxels inside the bounding box...?
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( args.flgCombineMaskAndBoundingBoxViaUnion )
  {
    typename ImageType::RegionType region = inImage->GetLargestPossibleRegion();

    region.SetSize( size );
    region.SetIndex( idxStart );

    IteratorType inputIterator( maskImage, region );

    for ( inputIterator.GoToBegin(); 
	  ! inputIterator.IsAtEnd();
	  ++inputIterator )
    {
      inputIterator.Set( 1 );
    }
  }
  

  // ...or only where the mask and bounding box overlap?

  else
  {
    bool flgInsideBoundingBox;
    typename ImageType::IndexType idx;
    typename ImageType::RegionType region = inImage->GetLargestPossibleRegion();

    IteratorType inputIterator( maskImage, region );

    for ( inputIterator.GoToBegin(); 
	  ! inputIterator.IsAtEnd();
	  ++inputIterator )
    {
      if ( inputIterator.Get( ) )
      {
	flgInsideBoundingBox = true;

	idx = inputIterator.GetIndex();

	for ( i=0; i<Dimension; i++ )
	  if ( ( idx[i] < idxStart[i] ) || ( idx[i] > idxEnd[i] ) )
	  {
	    flgInsideBoundingBox = false;
	    break;
	  }

	if ( flgInsideBoundingBox ) 
	  inputIterator.Set( 1 );
	else
	  inputIterator.Set( 0 );
      }
    }
  }


  // Mask the image
  // ~~~~~~~~~~~~~~

  typename MaskFilterType::Pointer filter = MaskFilterType::New();  
  
  filter->SetInput1( maskImage  );
  filter->SetInput2( inImage );
  


  // Write the masked image to a file
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typename OutputImageWriterType::Pointer imageWriter = OutputImageWriterType::New();

  imageWriter->SetFileName( args.fileOutputImage );
  imageWriter->SetInput( filter->GetOutput() );
  
  try
  {
    std::cout << "Writing masked image to: " << args.fileOutputImage << std::endl; 
    imageWriter->Update(); 
  }
  catch( itk::ExceptionObject & err ) 
  { 
    std::cerr << "ERROR: Failed to mask and write the image to a file." << err << std::endl; 
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
      result = MaskImage<2, unsigned char>( args );
      break;
      
    case itk::ImageIOBase::CHAR:
      result = MaskImage<2, char>( args );
      break;
      
    case itk::ImageIOBase::USHORT:
      result = MaskImage<2, unsigned short>( args );
      break;
      
    case itk::ImageIOBase::SHORT:
      result = MaskImage<2, short>( args );
      break;
      
    case itk::ImageIOBase::UINT:
      result = MaskImage<2, unsigned int>( args );
      break;
      
    case itk::ImageIOBase::INT:
      result = MaskImage<2, int>( args );
      break;
      
    case itk::ImageIOBase::ULONG:
      result = MaskImage<2, unsigned long>( args );
      break;
      
    case itk::ImageIOBase::LONG:
      result = MaskImage<2, long>( args );
      break;
      
    case itk::ImageIOBase::FLOAT:
      result = MaskImage<2, float>( args );
      break;
      
    case itk::ImageIOBase::DOUBLE:
      result = MaskImage<2, double>( args );
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
      result = MaskImage<3, unsigned char>( args );
      break;
      
    case itk::ImageIOBase::CHAR:
      result = MaskImage<3, char>( args );
      break;
      
    case itk::ImageIOBase::USHORT:
      result = MaskImage<3, unsigned short>( args );
      break;
      
    case itk::ImageIOBase::SHORT:
      result = MaskImage<3, short>( args );
      break;
      
    case itk::ImageIOBase::UINT:
      result = MaskImage<3, unsigned int>( args );
      break;
      
    case itk::ImageIOBase::INT:
      result = MaskImage<3, int>( args );
      break;
      
    case itk::ImageIOBase::ULONG:
      result = MaskImage<3, unsigned long>( args );
      break;
      
    case itk::ImageIOBase::LONG:
      result = MaskImage<3, long>( args );
      break;
      
    case itk::ImageIOBase::FLOAT:
      result = MaskImage<3, float>( args );
      break;
      
    case itk::ImageIOBase::DOUBLE:
      result = MaskImage<3, double>( args );
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
      result = MaskImage<4, unsigned char>( args );
      break;
      
    case itk::ImageIOBase::CHAR:
      result = MaskImage<4, char>( args );
      break;
      
    case itk::ImageIOBase::USHORT:
      result = MaskImage<4, unsigned short>( args );
      break;
      
    case itk::ImageIOBase::SHORT:
      result = MaskImage<4, short>( args );
      break;
      
    case itk::ImageIOBase::UINT:
      result = MaskImage<4, unsigned int>( args );
      break;
      
    case itk::ImageIOBase::INT:
      result = MaskImage<4, int>( args );
      break;
      
    case itk::ImageIOBase::ULONG:
      result = MaskImage<4, unsigned long>( args );
      break;
      
    case itk::ImageIOBase::LONG:
      result = MaskImage<4, long>( args );
      break;
      
    case itk::ImageIOBase::FLOAT:
      result = MaskImage<4, float>( args );
      break;
      
    case itk::ImageIOBase::DOUBLE:
      result = MaskImage<4, double>( args );
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
