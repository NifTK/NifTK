/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <math.h>
#include <float.h>

#include <niftkConversionUtils.h>
#include <niftkCommandLineParser.h>
#include <itkCommandLineHelper.h>
#include <itkCommandLineHelper.h>

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImage.h>
#include <itkImageRegionIterator.h>
#include <itkImageRegionConstIteratorWithIndex.h>

#include <itkNLMFilter.h>

#include <niftkDenoiseCLP.h>



// -------------------------------------------------------------------------
// Arguments
// -------------------------------------------------------------------------

class Arguments
{
public:
  float iSigma;
  std::vector<int> iRadiusSearch;
  std::vector<int> iRadiusComp;
  float iH;
  float iPs;

  std::string fileInputImage;
  std::string fileOutputImage;

  Arguments() {

    iSigma = 25.0f;
    iRadiusSearch.push_back(5);
    iRadiusSearch.push_back(5);
    iRadiusSearch.push_back(5);
    iRadiusComp.push_back(2);
    iRadiusComp.push_back(2);
    iRadiusComp.push_back(2);
    iH = 1.2f;
    iPs = 2.3f;
  }

  void Print(void) {
    std::cout << std::endl;

    std::cout << "ARG: Noise power: " << iSigma << std::endl;
    std::cout << "ARG: Search radius: [" << iRadiusSearch[0] <<","<< iRadiusSearch[1] <<","<< iRadiusSearch[2] <<"]"<< std::endl;
    std::cout << "ARG: Comparison radius: [" << iRadiusComp[0] <<","<< iRadiusComp[1] <<","<< iRadiusComp[2] <<"]"<< std::endl;
    std::cout << "ARG: h parameter: " << iH << std::endl;
    std::cout << "ARG: Preselection threshold: " << iPs << std::endl;

    std::cout << "ARG: Input image file: " << fileInputImage << std::endl;
    std::cout << "ARG: Output corrected image: " << fileOutputImage << std::endl;
    
    std::cout << std::endl;
  }
};


// --------------------------------------------------------------------------
// WriteImageToFile()
// --------------------------------------------------------------------------

template <class ImageType>
bool WriteImageToFile( std::string &fileOutput, const char *description,
                       typename ImageType::Pointer image )
{
  if ( fileOutput.length() ) {

    std::string fileModifiedOutput;

    typedef itk::ImageFileWriter< ImageType > FileWriterType;

    typename FileWriterType::Pointer writer = FileWriterType::New();

    writer->SetFileName( fileOutput );
    writer->SetInput( image );

    std::cout << "Writing " << description << " to file: "
	      << fileOutput << std::endl;
    writer->Update();

    return true;
  }
  else
    return false;
}


// -------------------------------------------------------------------------
// DoMain(Arguments args)
// -------------------------------------------------------------------------

template <int Dimension, class OutputPixelType>
int DoMain(Arguments &args)
{
    // do the typedefs
    typedef float InputPixelType;
    typedef itk::Image<InputPixelType,Dimension> ImageType;
    typename itk::ImageFileReader<ImageType>::Pointer reader = itk::ImageFileReader<ImageType>::New();
    reader->SetFileName( args.fileInputImage );

    try
    {
        reader->Update();
    }
    catch ( itk::ExceptionObject & e )
    {
        std::cerr << "exception in file reader " << std::endl;
        std::cerr << e.GetDescription() << std::endl;
        std::cerr << e.GetLocation() << std::endl;
        return EXIT_FAILURE;
    }


    typedef itk::NLMFilter< ImageType, ImageType > FilterType;
    typename FilterType::Pointer filter = FilterType::New();
    filter->SetInput( reader->GetOutput() );

    /** SET PARAMETERS TO THE FILTER */
    // The power of noise:
    filter->SetSigma( args.iSigma );
    // The search radius
    typename FilterType::InputImageSizeType radius;
    for( unsigned int d=0; d<Dimension; ++d )
        radius[d] = args.iRadiusSearch[d];
    filter->SetRSearch( radius );
    // The comparison radius:
    for( unsigned int d=0; d<Dimension; ++d )
        radius[d] = args.iRadiusComp[d];
    filter->SetRComp( radius );
    // The "h" parameter:
    filter->SetH( args.iH );
    // The preselection threshold:
    filter->SetPSTh( args.iPs );

    // Run the filter:
    try
    {
        filter->Update();
    }
    catch ( itk::ExceptionObject & e )
    {
        std::cerr << "exception in filter" << std::endl;
        std::cerr << e.GetDescription() << std::endl;
        std::cerr << e.GetLocation() << std::endl;
        return EXIT_FAILURE;
    }

    // Generate output image
    typename itk::ImageFileWriter<ImageType>::Pointer writer = itk::ImageFileWriter<ImageType>::New();
    writer->SetInput( filter->GetOutput() );
    writer->SetFileName( args.fileOutputImage );
    try
    {
        writer->Update();
    }
    catch ( itk::ExceptionObject & e )
    {
        std::cerr << "exception in file writer " << std::endl;
        std::cerr << e.GetDescription() << std::endl;
        std::cerr << e.GetLocation() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}


// -------------------------------------------------------------------------
// main( int argc, char *argv[] )
// -------------------------------------------------------------------------

int main( int argc, char *argv[] )
{
  // To pass around command line args
  Arguments args;

  // Validate command line args
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~

  PARSE_ARGS;

  args.iSigma                       = iSigma;
  args.iRadiusSearch                = iRadiusSearch;
  args.iRadiusComp                  = iRadiusComp;
  args.iH                           = iH;
  args.iPs                          = iPs;

  args.fileInputImage      = fileInputImage;
  args.fileOutputImage     = fileOutputImage;
  
  if ( args.fileInputImage.length() == 0 || args.fileOutputImage.length() == 0 )
  {
    commandLine.getOutput()->usage(commandLine);
    return EXIT_FAILURE;
  }

  int dims = itk::PeekAtImageDimensionFromSizeInVoxels(args.fileInputImage);
  if (dims != 3)
  {
    std::cout << "ERROR: Unsupported image dimension, image must be 3D" << std::endl;
    return EXIT_FAILURE;
  }

  args.Print();

  int result;

  switch (itk::PeekAtComponentType(args.fileInputImage))
  {
  case itk::ImageIOBase::UCHAR:
    result = DoMain<3, unsigned char>(args);
    break;

  case itk::ImageIOBase::CHAR:
    result = DoMain<3, char>(args);
    break;

  case itk::ImageIOBase::USHORT:
    result = DoMain<3, unsigned short>(args);
    break;

  case itk::ImageIOBase::SHORT:
    result = DoMain<3, short>(args);
    break;

  case itk::ImageIOBase::UINT:
    result = DoMain<3, unsigned int>(args);
    break;

  case itk::ImageIOBase::INT:
    result = DoMain<3, int>(args);
    break;

  case itk::ImageIOBase::ULONG:
    result = DoMain<3, unsigned long>(args);
    break;

  case itk::ImageIOBase::LONG:
    result = DoMain<3, long>(args);
    break;

  case itk::ImageIOBase::FLOAT:
    result = DoMain<3, float>(args);
    break;

  case itk::ImageIOBase::DOUBLE:
    result = DoMain<3, double>(args);
    break;

  default:
    std::cerr << "ERROR: Unsupported pixel format" << std::endl;
    return EXIT_FAILURE;
  }
  return result;
}
