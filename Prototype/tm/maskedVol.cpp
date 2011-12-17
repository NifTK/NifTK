/*=========================================================================

 This program takes as an input an  MRI volume and a mask.
 It returns the masked volume.

=========================================================================*/

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionConstIterator.h"
#include "itkImageRegionIterator.h"
//#include "itkGiplImageIO.h"

using namespace std;

int main( int argc, char ** argv )
{
   if( argc < 5 ) 
   { 
     std::cerr << "Usage: " << std::endl;
     std::cerr << argv[0] << " inputVolume maskVolume outputVolume maskValue(will mask everything greater than this value)";
     std::cerr <<  std::endl;
     return EXIT_FAILURE;
   }
   
   const     unsigned int   Dimension = 3;
   typedef   float  InputPixelType;  //short
   typedef   float  OutputPixelType; //short
  
   typedef itk::Image< InputPixelType,  Dimension >   InputImageType;
   typedef itk::Image< OutputPixelType, Dimension >   OutputImageType;

   // reader and writer for the input and output images
   typedef itk::ImageFileReader< InputImageType >  ReaderType;
   typedef itk::ImageFileWriter< OutputImageType >  WriterType;

   ReaderType::Pointer reader = ReaderType::New();
   ReaderType::Pointer readerMask = ReaderType::New();
   WriterType::Pointer writer = WriterType::New();

   reader->SetFileName( argv[1] );
   readerMask->SetFileName( argv[2] ); 
   writer->SetFileName( argv[3] );
   
   int maskValue = atoi( argv[4] );

   // the input and output images
   InputImageType::Pointer inputImage = InputImageType::New(); 
   InputImageType::Pointer maskImage = InputImageType::New();
   OutputImageType::Pointer outputImage = OutputImageType::New(); 
   
   reader->Update();
   readerMask->Update();

   inputImage = reader->GetOutput();
   maskImage = readerMask->GetOutput();
   
   // set the region for the input and the uotput images 
   InputImageType::RegionType inputRegion;
   inputRegion.SetSize( inputImage->GetLargestPossibleRegion().GetSize() );
   InputImageType::RegionType::IndexType inputStart;
   inputStart[0] = 0;
   inputStart[1] = 0;
   inputStart[2] = 0;
   inputRegion.SetIndex( inputStart );
 
   InputImageType::RegionType maskRegion;
   maskRegion.SetSize( maskImage->GetLargestPossibleRegion().GetSize() );
   InputImageType::RegionType::IndexType maskStart;
   maskStart[0] = 0;
   maskStart[1] = 0;
   maskStart[2] = 0;
   maskRegion.SetIndex( maskStart );
  
   OutputImageType::RegionType outputRegion;
   outputRegion.SetSize( inputImage->GetLargestPossibleRegion().GetSize() );
   OutputImageType::RegionType::IndexType outputStart;
   outputStart[0] = 0;
   outputStart[1] = 0;
   outputStart[2] = 0;
   outputRegion.SetIndex( outputStart );
   
   
   outputImage->SetRegions( outputRegion );

   outputImage->SetOrigin( inputImage->GetOrigin() ); 
   outputImage->SetSpacing( inputImage->GetSpacing() );
   outputImage->Allocate();

   // Iterators declaration
   typedef itk::ImageRegionConstIterator< InputImageType > ConstIteratorType;
   typedef itk::ImageRegionIterator< OutputImageType > IteratorType;

   ConstIteratorType in( inputImage, inputImage->GetLargestPossibleRegion() );
   ConstIteratorType mask( maskImage, maskImage->GetLargestPossibleRegion() );
   IteratorType out( outputImage, inputImage->GetLargestPossibleRegion() );
   
   /*std::cout << "Input volume origin" << reader->GetOutput()->GetOrigin() << std::endl;
   std::cout << "Mask volume origin" << readerMask->GetOutput()->GetOrigin() << std::endl;
   std::cout << "Input volume resolution" << reader->GetOutput()->GetSpacing() << std::endl;
   std::cout << "Mask volume resolution" << readerMask->GetOutput()->GetSpacing() << std::endl;
   std::cout << "Input volume size" << reader->GetOutput()->GetLargestPossibleRegion().GetSize() << std::endl;
   std::cout << "Mask volume size" << readerMask->GetOutput()->GetLargestPossibleRegion().GetSize() << std::endl;*/

   for ( in.GoToBegin(), mask.GoToBegin(), out.GoToBegin(); !in.IsAtEnd(); ++in, ++mask, ++out )
   {
     if ( mask.Get()>maskValue )
       out.Set( in.Get() );
     else
       out.Set( 0 ); 
   }
   
   // write the output
   writer->SetInput( outputImage );

   try 
   { 
     std::cout << "Writing output image... " << std::endl;
     writer->Update();
   } 
   catch( itk::ExceptionObject & err ) 
   { 
     std::cerr << "ERROR: ExceptionObject caught !" << std::endl; 
     std::cerr << err << std::endl; 
   } 
   
   return 0;
}


