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
#include <niftkCommandLineParser.h>
#include <itkCommandLineHelper.h>

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkExtractImageFilter.h>

#include <boost/filesystem.hpp>


/*!
 * \file niftkSplitVolumeIntoVoxelPlanes.cxx
 * \page niftkSplitVolumeIntoVoxelPlanes
 * \section niftkSplitVolumeIntoVoxelPlanesSummary Runs ITK ExtractImageFilter on every plane of voxels in the input volume.
 */

struct niftk::CommandLineArgumentDescription clArgList[] = {

  {OPT_SWITCH, "v", NULL, "Verbose output."},
  {OPT_SWITCH, "dbg", NULL, "Output debugging info."},

  {OPT_INT, "slice", "dimension", "The dimension along which the volume is sliced (x=0, y=1, z=2) [2]."},

  {OPT_STRING|OPT_REQ, "o", "fileOut", "The file name of the output voxel planes."},

  {OPT_STRING|OPT_LONELY|OPT_REQ, NULL, "fileIn", "Input image volume."},

  {OPT_DONE, NULL, NULL,
   "Program to split an image volume into individual planes of voxels.\n"
  }
};

enum {
  O_VERBOSE,
  O_DEBUG,

  O_SLICE_DIMENSION,

  O_OUTPUT_FILE,
  O_INPUT_VOLUME
};


struct arguments
{
  bool flgVerbose;
  bool flgDebug;

  int sliceDimension;

  std::string fileOutput;
  std::string fileInput;

  arguments() {
    flgVerbose = false;
    flgDebug = false;

    sliceDimension = 2;
  }
};


std::string AddSuffix( std::string filename, unsigned int iSlice ) 
{
  char strSlice[128];

  boost::filesystem::path pathname( filename );
  boost::filesystem::path ofilename;

  std::string extension = pathname.extension().string();
  std::string stem = pathname.stem().string();

  if ( extension == std::string( ".gz" ) ) {

    extension = pathname.stem().extension().string() + extension;
    stem = pathname.stem().stem().string();
  }

  sprintf(strSlice, "_%04d", iSlice);

  ofilename = pathname.parent_path() /
    boost::filesystem::path( stem + std::string( strSlice ) + extension );
    
  return ofilename.string();
}



template <int InputDimension, class PixelType>
int DoMain(arguments args)
{
  const   unsigned int OutputDimension = InputDimension - 1;

  unsigned int iSlice;
  unsigned int nSlices;

  std::string fileOutputSlice;

  typedef itk::Image< PixelType, InputDimension >                  InputImageType;
  typedef itk::Image< PixelType, OutputDimension >                 OutputImageType; 

  typedef itk::ImageFileReader< InputImageType >                   InputImageReaderType;
  typedef itk::ImageFileWriter< OutputImageType >                  OutputImageWriterType;

  typedef itk::ExtractImageFilter<InputImageType, OutputImageType> ExtractImageFilterType;

  typename InputImageType::SizeType size;
  typename InputImageType::IndexType index;
  typename InputImageType::RegionType region;
  
  typename InputImageReaderType::Pointer imageReader = InputImageReaderType::New();


  // Read the input image
  // ~~~~~~~~~~~~~~~~~~~~

  imageReader->SetFileName(args.fileInput);

  try {
    imageReader->Update();
  }
  catch( itk::ExceptionObject & err )
  {
    std::cerr << "ERROR: Failed to read image: " << args.fileInput << std::endl
              << err << std::endl;
    return EXIT_FAILURE;
  }

  if ( args.flgDebug )
  {
    imageReader->GetOutput()->Print( std::cout );
  }

  typename InputImageType::Pointer image = imageReader->GetOutput();

  image->DisconnectPipeline();
  
  region = image->GetLargestPossibleRegion();
  size = region.GetSize();
  index = region.GetIndex();

  if ( args.flgVerbose )
  {
    std::cout << std::endl
              << "Input image region is: " << std::endl
              << region << std::endl;
  }

  nSlices = size[ args.sliceDimension ];

  size[ args.sliceDimension ] = 0;
  region.SetSize(size);

  
  for ( iSlice=0; iSlice<nSlices; iSlice++ )
  {
  
    index[ args.sliceDimension ] = iSlice;

    region.SetIndex(index);
  
    if ( args.flgVerbose )
    {
      std::cout << std::endl
                << "Region for slice: " << iSlice << " is " << std::endl
                << region << std::endl;
    }

    typename ExtractImageFilterType::Pointer filter = ExtractImageFilterType::New();

    filter->SetInput( image );
    filter->SetExtractionRegion( region );
    filter->SetDirectionCollapseToIdentity();

    typename OutputImageWriterType::Pointer imageWriter = OutputImageWriterType::New();

    fileOutputSlice = AddSuffix( args.fileOutput, iSlice);

    imageWriter->SetFileName( fileOutputSlice );
    imageWriter->SetInput( filter->GetOutput() );
  
    try
    {
      std::cout << "Writing slice " << iSlice << " to file: " << fileOutputSlice << std::endl;
      imageWriter->Update(); 
    }
    catch( itk::ExceptionObject & err ) 
    { 
      std::cerr << "Failed: " << err << std::endl; 
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
  struct arguments args;

  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);

  CommandLineOptions.GetArgument( O_VERBOSE, args.flgVerbose );
  CommandLineOptions.GetArgument( O_DEBUG, args.flgDebug );

  CommandLineOptions.GetArgument( O_SLICE_DIMENSION, args.sliceDimension );

  CommandLineOptions.GetArgument( O_OUTPUT_FILE, args.fileOutput );

  CommandLineOptions.GetArgument(O_INPUT_VOLUME, args.fileInput );

  
  std::cout << "Slice dimension: " << args.sliceDimension << std::endl
            << "Input image:     " << args.fileInput << std::endl
            << "Output image:    " << args.fileOutput << std::endl;


  int dims = itk::PeekAtImageDimensionFromSizeInVoxels( args.fileInput );
  if (dims != 3)
  {
    std::cout << "ERROR: Input image must be 3D" << std::endl;
    return EXIT_FAILURE;
  }
   
  if (( args.sliceDimension < 0 ) || ( args.sliceDimension >= dims ))
  {
    std::cout << "ERROR: Slice dimension (" << args.sliceDimension 
              << ") is outside permitted range: 0 to " << dims - 1 << std::endl;
    return EXIT_FAILURE;
  }
   

  int result;

  switch ( itk::PeekAtComponentType( args.fileInput ) )
  {

  case itk::ImageIOBase::UCHAR:
    std::cout << "Input is UNSIGNED CHAR" << std::endl;
    result = DoMain<3, unsigned char>( args );  
    break;

  case itk::ImageIOBase::CHAR:
    std::cout << "Input is CHAR" << std::endl;
    result = DoMain<3, char>( args );  
    break;

  case itk::ImageIOBase::USHORT:
    std::cout << "Input is UNSIGNED SHORT" << std::endl;
    result = DoMain<3, unsigned short>( args );  
    break;

  case itk::ImageIOBase::SHORT:
    std::cout << "Input is SHORT" << std::endl;
    result = DoMain<3, short>( args );  
    break;

  case itk::ImageIOBase::UINT:
    std::cout << "Input is UNSIGNED INT" << std::endl;
    result = DoMain<3, unsigned int>( args );  
    break;

  case itk::ImageIOBase::INT:
    std::cout << "Input is INT" << std::endl;
    result = DoMain<3, int>( args );  
    break;

  case itk::ImageIOBase::ULONG:
    std::cout << "Input is UNSIGNED LONG" << std::endl;
    result = DoMain<3, unsigned long>( args );  
    break;

  case itk::ImageIOBase::LONG:
    std::cout << "Input is LONG" << std::endl;
    result = DoMain<3, long>( args );  
    break;

  case itk::ImageIOBase::FLOAT:
    std::cout << "Input is FLOAT" << std::endl;
    result = DoMain<3, float>( args );  
    break;

  case itk::ImageIOBase::DOUBLE:
    std::cout << "Input is DOUBLE" << std::endl;
    result = DoMain<3, double>( args );  
    break;

  default:
    std::cerr << "ERROR: non standard pixel format" << std::endl;
    return EXIT_FAILURE;
  }

  return result;
}
