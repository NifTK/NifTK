/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/


#include <itkLogHelper.h>
#include <niftkConversionUtils.h>
#include <itkCommandLineHelper.h>

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkRelabelComponentImageFilter.h>
#include <itkImageRegionIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkScalarConnectedComponentImageFilter.h>

#include <niftkScalarConnectedComponentImageFilterCLP.h>


/*!
 * \file niftkScalarConnectedComponentImageFilter.cxx
 * \page niftkScalarConnectedComponentImageFilter
 * \section niftkScalarConnectedComponentImageFilterSummary Identifies connected components in and image and assigns each a unique label. 
 *
 * This program uses ScalarConnectedComponentImageFilter to label the connected objects in an input image. Two adjacent voxels are considered members of the same object if they are within a specified intensity distance of each other. The labels are ordered according to the size of the objects.
 *
 * \li Dimensions: 2,3.
 * \li Pixel type: Scalars only of unsigned char, char, unsigned short, short, unsigned int, int, unsigned long, long, float and double.
 *
 * \section niftkScalarConnectedComponentImageFilterCaveats Caveats
 * \li None
 */

struct arguments
{
  bool flgLargestObject;

  float distanceThreshold;
  float minSize;

  float minLabelRank;
  float maxLabelRank;

  float border;

  std::string inputImage;
  std::string outputImage;

  arguments() {
    distanceThreshold = 0.;
    minSize = 20;

    minLabelRank = 0.;
    maxLabelRank = 0.;

    border = 0;
  }
  
};

template <int Dimension, class PixelType> 
int DoMain(arguments args)
{  
  typedef typename itk::Image< PixelType, Dimension >     InputImageType;   
  typedef typename itk::ImageFileReader< InputImageType > InputImageReaderType;

  typedef unsigned int LabelPixelType;
  typedef itk::Image<LabelPixelType, Dimension > LabelImageType;

  typedef typename itk::ImageFileWriter< LabelImageType > OutputImageWriterType;

  
  // Read the input image
  // ~~~~~~~~~~~~~~~~~~~~

  typename InputImageReaderType::Pointer imageReader = InputImageReaderType::New();
  imageReader->SetFileName( args.inputImage );
  
  try
  {
    std::cout << "Reading input image: " << args.inputImage << std::endl; 
    imageReader->Update(); 
  }
  catch( itk::ExceptionObject & err ) 
  { 
    std::cerr << std::endl << "ERROR: Failed to read the input image: " << err
	      << std::endl << std::endl; 
    return EXIT_FAILURE;
  }                

  typename InputImageType::Pointer image = imageReader->GetOutput();
  image->DisconnectPipeline();


  // Ignore the image border

  if ( args.border )
  {
    int i;

    typename InputImageType::RegionType region = image->GetLargestPossibleRegion();
    typename InputImageType::SizeType size = region.GetSize();

    typename InputImageType::IndexType start = region.GetIndex();

    for ( i=0; i<Dimension; i++ )
    {
      start[i] += static_cast<int>( args.border );
      size[i]  -= static_cast<int>( 2.*args.border );
    }

    region.SetSize( size );
    region.SetIndex( start );

    typedef itk::ImageRegionIteratorWithIndex< InputImageType > InputIteratorType;
  
    InputIteratorType itImage( image, image->GetLargestPossibleRegion() );

    typename InputImageType::IndexType idx;
    
    itImage.GoToBegin();
    while (! itImage.IsAtEnd() ) 
    {
      idx = itImage.GetIndex();

      if ( ! region.IsInside( idx ) )
      {
        itImage.Set( 0.);
      }
      
      ++itImage;
    }
  }


  // Detect connected components
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef itk::ScalarConnectedComponentImageFilter < InputImageType, LabelImageType >
    ConnectedComponentImageFilterType;

  typename ConnectedComponentImageFilterType::Pointer connected =
    ConnectedComponentImageFilterType::New ();

  connected->SetInput( image );
  connected->SetDistanceThreshold( args.distanceThreshold );

  // Relabel the object by size

  typedef itk::RelabelComponentImageFilter <LabelImageType, LabelImageType >
    RelabelFilterType;

  typename RelabelFilterType::Pointer relabel = RelabelFilterType::New();

  relabel->SetMinimumObjectSize( args.minSize );

  relabel->SetInput( connected->GetOutput() );

  try
  {
    relabel->Update(); 
  }
  catch( itk::ExceptionObject & err ) 
  { 
    std::cerr << "Failed: " << err << std::endl; 
    return EXIT_FAILURE;
  }                

  // Set the label range to be keep i.e. set labels outside this range to zero

  typename LabelImageType::Pointer imLabels = relabel->GetOutput();
  imLabels->DisconnectPipeline();

  typedef itk::ImageRegionIterator< LabelImageType > LabelIteratorType;
  
  LabelIteratorType itImage( imLabels, imLabels->GetLargestPossibleRegion() );
    
  itImage.GoToBegin();

  while (! itImage.IsAtEnd() ) 
  {
    if ( ( itImage.Get() < args.minLabelRank ) || 
         ( itImage.Get() > args.maxLabelRank ) )
    {
      itImage.Set( 0.);
    }
    
    ++itImage;
  }
  
  relabel->SetInput( imLabels );

  try
  {
    relabel->Update(); 

    std::cout << "Number of connected objects: " << relabel->GetNumberOfObjects() << std::endl << std::endl
              << "Size of largest object: " << relabel->GetSizeOfObjectsInPixels()[0] << std::endl << std::endl;
  }
  catch( itk::ExceptionObject & err ) 
  { 
    std::cerr << "Failed: " << err << std::endl; 
    return EXIT_FAILURE;
  }                


  // Extract the largest object?

  if ( args.flgLargestObject )
  {
    relabel->SetMinimumObjectSize( relabel->GetSizeOfObjectsInPixels()[0] );
  }

  
  // Write the output image
  // ~~~~~~~~~~~~~""~~~~~~~
  
  typename OutputImageWriterType::Pointer imageWriter = OutputImageWriterType::New();

  imageWriter->SetFileName( args.outputImage );
  imageWriter->SetInput( relabel->GetOutput() );
  
  try
  {
    std::cout << "Writing labelled image: " << args.outputImage << std::endl; 
    imageWriter->Update(); 
  }
  catch( itk::ExceptionObject & err ) 
  { 
    std::cerr << "Failed: " << err << std::endl; 
    return EXIT_FAILURE;
  }                

  return EXIT_SUCCESS;
}


/**
 *  \brief Identifies connected components in and image and assigns each a unique label. 
 */

int main(int argc, char** argv)
{
  // To pass around command line args
  PARSE_ARGS;

  // To pass around command line args
  struct arguments args;

  args.flgLargestObject = flgLargestObject;

  args.distanceThreshold = distanceThreshold;
  args.minSize = minSize;

  args.minLabelRank = minLabelRank;
  args.maxLabelRank = maxLabelRank;

  args.border = border;

  args.inputImage=inputImage.c_str();
  args.outputImage=outputImage.c_str();

  std::cout << "Input image:  " << args.inputImage << std::endl
            << "Output image: " << args.outputImage << std::endl;

  std::cout << std::endl
            << "Connected object intensity difference threshold: " << args.distanceThreshold << std::endl
            << "Min object size: " << args.minSize << std::endl
            << "Border width: " << args.border << std::endl;

  if ( args.minLabelRank || args.maxLabelRank )
    std::cout << "Object size rank range to be keep (1=largest): " << args.minLabelRank 
              << " to " << args.maxLabelRank << std::endl;


  // Validate command line args
  if (args.inputImage.length() == 0 ||
      args.outputImage.length() == 0)
  {
    return EXIT_FAILURE;
  }

  int dims = itk::PeekAtImageDimensionFromSizeInVoxels(args.inputImage);
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
   
  int result;

  switch (itk::PeekAtComponentType(args.inputImage))
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
