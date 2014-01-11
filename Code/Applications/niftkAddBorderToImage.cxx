/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

/*!
 * \file niftkAddBorderToImage.cxx
 * \page niftkAddBorderToImage
 * \section niftkAddBorderToImageSummary Adds a fixed width border (specified in millimetres) to an image. The border intensity can be specified explicitly but the default action is to use the intensity of the first voxel.
 *
 * \li Pixel type: Scalars only.
 */


#include <niftkConversionUtils.h>

#include <itkLogHelper.h>
#include <itkCommandLineHelper.h>
#include <itkMath.h>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkNumericTraits.h>
#include <itkImageRegionIteratorWithIndex.h>

#include <niftkAddBorderToImageCLP.h>


//  -------------------------------------------------------------------------
//  arguments
//  -------------------------------------------------------------------------

struct arguments
{
  bool flgUserSetBorderValue;

  std::vector<double> width;
  double intensity;

  std::string fileInputImage;
  std::string fileOutputImage;

  arguments() {
    flgUserSetBorderValue = false;
    intensity = 0.;
  }
};



//  -------------------------------------------------------------------------
//  AddBorderToImage()
/// \brief Adds a border to the image
//  -------------------------------------------------------------------------

template <int Dimension, class PixelType>
int AddBorderToImage( arguments &args )
{

  typedef itk::Image< PixelType, Dimension > ImageType;   
  typedef itk::ImageFileReader< ImageType >  InputImageReaderType;
  typedef itk::ImageFileWriter< ImageType >  OutputImageWriterType;

  typedef itk::ImageRegionIteratorWithIndex< ImageType > IteratorType;    

  typename ImageType::Pointer inImage = 0;
  typename ImageType::Pointer outImage = 0;


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


  // Allocate the output image
  // ~~~~~~~~~~~~~~~~~~~~~~~~~

  unsigned int i;

  typename ImageType::SpacingType spacing = inImage->GetSpacing();
  typename ImageType::IndexType   start;

  typename ImageType::PointType   inOrigin  = inImage->GetOrigin();
  typename ImageType::RegionType  inRegion  = inImage->GetLargestPossibleRegion();

  typename ImageType::SizeType    inSize    = inRegion.GetSize();
  
  if ( ! args.flgUserSetBorderValue )
  {
    typename ImageType::IndexType index;
    for (i=0; i < Dimension; i++)
    {
      index[i] = 0;  
    }
    args.intensity = inImage->GetPixel( index );
  }

  std::cout << "Setting border region to: " + niftk::ConvertToString( args.intensity ) << std::endl;

  typename ImageType::PointType   outOrigin;
  typename ImageType::RegionType  outRegion;

  typename ImageType::SizeType    outSize;

  double widthInMillimetres;
  double widthInVoxels;

  outSize = inSize;

  std::cout << "Expanding image by: ";
  for (i=0; i<Dimension; i++)
  {
    widthInVoxels = itk::Math::Round<double>( args.width[i]/spacing[i] );
    widthInMillimetres = widthInVoxels*spacing[i];

    outOrigin[i] = inOrigin[i] - widthInMillimetres;

    outSize[i] = inSize[i] + 2*static_cast<itk::SizeValueType>( widthInVoxels );

    start[i] = widthInVoxels;

    std::cout << widthInVoxels;
    if ( i < Dimension - 1 ) std::cout << ", ";
  }
  std::cout << std::endl;

  outRegion.SetSize( outSize );

  outImage = ImageType::New();

  outImage->SetSpacing( spacing );
  outImage->SetRegions( outRegion );
  outImage->SetOrigin(  outOrigin );

  outImage->Allocate();
  outImage->FillBuffer( static_cast<PixelType>( args.intensity ) );


  // Copy over the input intensities
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  outRegion.SetSize(  inSize );
  outRegion.SetIndex( start );

  IteratorType inputIterator(   inImage,  inRegion );
  IteratorType outputIterator( outImage, outRegion );
      
  std::cout << "Copying input region: " << inRegion
            << " to output region: " << outRegion << std::endl;

  for ( inputIterator.GoToBegin(), outputIterator.GoToBegin(); 
        (! inputIterator.IsAtEnd()) && (! outputIterator.IsAtEnd());
        ++inputIterator, ++outputIterator )
  {
    outputIterator.Set( inputIterator.Get() );
  }


  // Write the image to a file
  // ~~~~~~~~~~~~~~~~~~~~~~~~~

  typename OutputImageWriterType::Pointer imageWriter = OutputImageWriterType::New();

  imageWriter->SetFileName( args.fileOutputImage );
  imageWriter->SetInput( outImage );
  
  try
  {
    std::cout << "Writing output image to: " << args.fileOutputImage << std::endl; 
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
  unsigned int i;
  arguments args;
  

  // Validate command line args
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~

  PARSE_ARGS;

  if ( fileInputImage.length() == 0 || fileOutputImage.length() == 0 )
  {
    commandLine.getOutput()->usage(commandLine);
    return EXIT_FAILURE;
  }

  args.fileInputImage  = fileInputImage;
  args.fileOutputImage = fileOutputImage;

  int dims = itk::PeekAtImageDimensionFromSizeInVoxels( args.fileInputImage );
  
  // Get the border width if specified

  args.width.resize( dims );

  if ( width.size() == 0 )      // No value specified so use default width
  {
    args.width.assign( dims, 10. );
  }
  else if ( width.size() == 1 ) // One value: use for all dimensions
  {
    args.width.assign( dims, width[0] );
  }
  else                          // Different value of each axis
  {
    args.width.assign( dims, 0. );

    for ( i=0; (i<width.size()) && (i<args.width.size()); i++ )
    {
      args.width[i] = width[i];
    }
  }

  std::cout << "Border widths will be: ";
  for ( i=0; i<args.width.size(); i++ )
  {
    std::cout << args.width[i];

    if ( i < args.width.size() - 1 ) std::cout << ", ";
  }
  std::cout << std::endl;
  

  // Get the border intensity if specified

  if ( intensity.size() == 0 )
  {
    args.flgUserSetBorderValue = false;
    args.intensity = 0;
  }
  else if ( intensity.size() == 1 )
  {
    args.flgUserSetBorderValue = true;
    args.intensity = intensity[0];
  }
  else
  {
    std::cerr << "ERROR: The border image intensity (";
    unsigned int i;
    for (i=0; i<intensity.size(); i++)
    {
      std::cerr << intensity[i];
      if ( i + 1 < intensity.size() ) std::cerr << ",";
    }
    std::cerr << ") is not recognised" << std::endl;
    return( EXIT_FAILURE );
  }
  


  // Find the image dimension and the image type

  int result = 0;
  
  switch ( dims )
  {
  case 2: 
  {
    switch ( itk::PeekAtComponentType( args.fileInputImage ) )
    {
    case itk::ImageIOBase::UCHAR:
      result = AddBorderToImage<2, unsigned char>( args );
      break;
      
    case itk::ImageIOBase::CHAR:
      result = AddBorderToImage<2, char>( args );
      break;
      
    case itk::ImageIOBase::USHORT:
      result = AddBorderToImage<2, unsigned short>( args );
      break;
      
    case itk::ImageIOBase::SHORT:
      result = AddBorderToImage<2, short>( args );
      break;
      
    case itk::ImageIOBase::UINT:
      result = AddBorderToImage<2, unsigned int>( args );
      break;
      
    case itk::ImageIOBase::INT:
      result = AddBorderToImage<2, int>( args );
      break;
      
    case itk::ImageIOBase::ULONG:
      result = AddBorderToImage<2, unsigned long>( args );
      break;
      
    case itk::ImageIOBase::LONG:
      result = AddBorderToImage<2, long>( args );
      break;
      
    case itk::ImageIOBase::FLOAT:
      result = AddBorderToImage<2, float>( args );
      break;
      
    case itk::ImageIOBase::DOUBLE:
      result = AddBorderToImage<2, double>( args );
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
      result = AddBorderToImage<3, unsigned char>( args );
      break;
      
    case itk::ImageIOBase::CHAR:
      result = AddBorderToImage<3, char>( args );
      break;
      
    case itk::ImageIOBase::USHORT:
      result = AddBorderToImage<3, unsigned short>( args );
      break;
      
    case itk::ImageIOBase::SHORT:
      result = AddBorderToImage<3, short>( args );
      break;
      
    case itk::ImageIOBase::UINT:
      result = AddBorderToImage<3, unsigned int>( args );
      break;
      
    case itk::ImageIOBase::INT:
      result = AddBorderToImage<3, int>( args );
      break;
      
    case itk::ImageIOBase::ULONG:
      result = AddBorderToImage<3, unsigned long>( args );
      break;
      
    case itk::ImageIOBase::LONG:
      result = AddBorderToImage<3, long>( args );
      break;
      
    case itk::ImageIOBase::FLOAT:
      result = AddBorderToImage<3, float>( args );
      break;
      
    case itk::ImageIOBase::DOUBLE:
      result = AddBorderToImage<3, double>( args );
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
      result = AddBorderToImage<4, unsigned char>( args );
      break;
      
    case itk::ImageIOBase::CHAR:
      result = AddBorderToImage<4, char>( args );
      break;
      
    case itk::ImageIOBase::USHORT:
      result = AddBorderToImage<4, unsigned short>( args );
      break;
      
    case itk::ImageIOBase::SHORT:
      result = AddBorderToImage<4, short>( args );
      break;
      
    case itk::ImageIOBase::UINT:
      result = AddBorderToImage<4, unsigned int>( args );
      break;
      
    case itk::ImageIOBase::INT:
      result = AddBorderToImage<4, int>( args );
      break;
      
    case itk::ImageIOBase::ULONG:
      result = AddBorderToImage<4, unsigned long>( args );
      break;
      
    case itk::ImageIOBase::LONG:
      result = AddBorderToImage<4, long>( args );
      break;
      
    case itk::ImageIOBase::FLOAT:
      result = AddBorderToImage<4, float>( args );
      break;
      
    case itk::ImageIOBase::DOUBLE:
      result = AddBorderToImage<4, double>( args );
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
