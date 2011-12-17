/*=========================================================================


=========================================================================*/


#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkBinaryDilateImageFilter.h"
#include "itkBinaryBallStructuringElement.h"
#include "itkNeighborhoodAllocator.h"

int main( int argc, char * argv[] )
{
 if( argc < 4 ) 
 { 
   std::cerr << "Usage: " << std::endl;
   std::cerr << argv[0] << "  inputImageFile  outputImageFile dilateValue ";
   std::cerr <<  std::endl;
   return EXIT_FAILURE;
 }
 // imput and output decl  
 const int dimension = 3;
 typedef    short    InputPixelType; //float
 typedef    short    OutputPixelType; //double
 typedef itk::Image< InputPixelType,  dimension >   InputImageType;
 typedef itk::Image< OutputPixelType, dimension  >   OutputImageType; 
 typedef itk::NeighborhoodAllocator < InputImageType::PixelType > NeighType; 
 typedef itk::BinaryBallStructuringElement< InputImageType::PixelType, dimension, NeighType > KernelType;
 
 // reader and writer for the input and output images
 typedef itk::ImageFileReader< InputImageType >  ReaderType;
 typedef itk::ImageFileWriter< OutputImageType >  WriterType;

 ReaderType::Pointer reader = ReaderType::New();
 WriterType::Pointer writer = WriterType::New();
 reader->SetFileName( argv[1] );
 writer->SetFileName( argv[2] );
 
 // defs of transorm and filter
 typedef itk::BinaryDilateImageFilter< InputImageType, OutputImageType, KernelType > FilterType;
 FilterType::Pointer filter = FilterType::New();

 KernelType kernel;

 kernel.CreateStructuringElement();
 kernel.SetRadius( atoi(argv[3]) );
  
 //kernel.PrintSelf(std::cout);
 
 filter->SetKernel( kernel );
 filter->SetForegroundValue( 1 );
 filter->SetBackgroundValue( 0 ); 
 filter->SetDilateValue( 1 );
 
 OutputImageType::Pointer outputImage = OutputImageType::New();

 reader->Update();
 filter->SetInput( reader->GetOutput() );

 //filter->PrintSelf( std::cout );

 writer->SetInput( filter->GetOutput() );

 try
 {
	writer->Update();
 }
 catch( itk::ExceptionObject & excep )
 {
	std::cerr << "Exception caught !" << std::endl;
	std::cerr << excep << std::endl;
 }
 
 
}
