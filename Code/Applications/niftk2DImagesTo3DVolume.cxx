/*=========================================================================

  Program to compose a 3D volume from individual 2D images.  
  
=========================================================================*/

#include "niftkConversionUtils.h"
#include "niftkCommandLineParser.h"
#include "itkCommandLineHelper.h"

#include "itkImage.h"
#include "itkIndex.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionIterator.h"
#include "itkMinimumMaximumImageCalculator.h"
#include "itkRescaleIntensityImageFilter.h"

#include <itkVector.h>
#include <iostream>
#include <string>

using namespace std;

struct niftk::CommandLineArgumentDescription clArgList[] = {
  {OPT_STRING|OPT_REQ, "o", "string", "Output 3D image."},
  {OPT_FLOAT, "spZ", "float", "Spacing (mm) in the Z direction. [5]"},
  
  {OPT_STRING|OPT_LONELY|OPT_REQ, NULL, "filename", "Individual 2D images."},
  {OPT_MORE, NULL, "...", NULL},
   
  {OPT_DONE, NULL, NULL, 
   "Program to compose a 3D volume from individual 2D images.\n"
  }
};

enum { 
  O_FILE_OUTPUT,
  O_SPACING_Z,
  
  O_FILE_2D,
  O_MORE
};

int main( int argc, char * argv[] )
{

  std::string fileOutput;
 
  float spacingZ = 5;
 
  char *file2D = 0; // A mandatory character string argument
  char **file2Ds = 0; // Multiple character string arguments
 
  int arg; // Index of arguments in command line 
 
  unsigned int nImages = 0; 
 
  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);

  CommandLineOptions.GetArgument( O_FILE_OUTPUT, fileOutput );
  CommandLineOptions.GetArgument( O_SPACING_Z, spacingZ );
    
  CommandLineOptions.GetArgument( O_FILE_2D, file2D);

  // Call the 'OPT_MORE' option to determine the position of the list
  // of extra command line options ('arg').
  CommandLineOptions.GetArgument(O_MORE, arg);
  
  if (arg < argc)   // Many 2D Images
  {
    nImages = argc - arg + 1;
    file2Ds = &argv[arg-1];
    
    std::cout << "2D images: " << std::endl;
    for (unsigned int i=0; i<nImages; i++)
      std::cout << niftk::ConvertToString( (int) i+1) << " " << file2Ds[i] << std::endl;
  }
  else if (file2D)  // Single 2D Image
  {
    nImages = 1;
    file2Ds = &file2D;
    std::vector< float > angles( nImages );
    
    std::cout << "2D image: " << file2Ds[0] << std::endl;
  }
  else 
  {
    nImages = 0;
    file2Ds = 0;
   
    std::cout << "No input file specified. Exit." << std::endl;
   
    return 1;
  }
   
  std::cout << " -- " << std::endl;

  // input and output decl  
  const int dimension = 3;
  typedef  unsigned short PixelType;
  
  typedef itk::Image< PixelType, 2  >   InputImageType;     
  typedef itk::Image< PixelType, dimension  >   OutputImageType; 
  typedef itk::ImageFileWriter< OutputImageType >  WriterType;

  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( fileOutput );

  const unsigned int NUMINPUTIMAGES = nImages;
  
  std::vector<InputImageType::Pointer> images2D( NUMINPUTIMAGES );
  
  std::vector< float > angles( NUMINPUTIMAGES );
 
  for (unsigned int k=0; k< NUMINPUTIMAGES; ++k )
  {
  
    typedef   itk::ImageFileReader< InputImageType >  ReaderType;
    ReaderType::Pointer reader = ReaderType::New();
    std::cout << "Reading file: " << file2Ds[k] << " ... " << std::endl;  
    reader->SetFileName( file2Ds[k] );
    try
    {
      reader->Update();
    }
    catch( itk::ExceptionObject & exp )
    {
      std::cout << "Exception thrown while reading the input file.";
      std::cerr << exp << std::endl; 
      return EXIT_FAILURE;
    }
    images2D[k] = reader->GetOutput();

  }// end for
  
  
  // Composing one 3D volume consisting of the 2D images
  OutputImageType::Pointer outputImage = OutputImageType::New(); 
  
  OutputImageType::RegionType outputRegion;
  OutputImageType::RegionType::IndexType outputStart;
  outputStart[0] = 0;
  outputStart[1] = 0;
  outputStart[2] = 0;
  outputRegion.SetIndex( outputStart );
  
  OutputImageType::SizeType size;
  size[0] = images2D[0]->GetLargestPossibleRegion().GetSize()[0];
  size[1] = images2D[0]->GetLargestPossibleRegion().GetSize()[1];
  size[2] = NUMINPUTIMAGES;
  outputRegion.SetSize( size );   
  outputImage->SetRegions( outputRegion );

  OutputImageType::PointType origin;
  origin[0] = 0.;
  origin[1] = 0.;
  origin[2] = 0.;
  outputImage->SetOrigin( origin ); 
  
  OutputImageType::SpacingType spacing;
  spacing[0] = images2D[0]->GetSpacing()[0];
  spacing[1] = images2D[0]->GetSpacing()[1];
  spacing[2] = spacingZ;//5;
  outputImage->SetSpacing( spacing );
  
  outputImage->Allocate();

  typedef itk::ImageRegionIterator< OutputImageType > IteratorType;

  IteratorType out( outputImage, outputImage->GetLargestPossibleRegion() );
  
  itk::Index< 3 > tmpIndex;
  itk::Index< 2 > tmp2Dindex;
 
  
  std::cout << "Composing a 3D volume consisting of: " << NUMINPUTIMAGES 
            << " 2D images..." << std::endl; 

  for ( out.GoToBegin(); !out.IsAtEnd(); ++out )
  {
    tmpIndex = out.GetIndex();
    tmp2Dindex[0] = tmpIndex[0];
    tmp2Dindex[1] = tmpIndex[1];    

    out.Set( images2D[tmpIndex[2]]->GetPixel(tmp2Dindex) );
  }
  
  writer->SetInput( outputImage );

  try
  {
    writer->Update();
  }
  catch( itk::ExceptionObject & excep )
  {
    std::cerr << "Exception caught !" << std::endl;
    std::cerr << excep << std::endl;
  }
  
  return 0;
}
































