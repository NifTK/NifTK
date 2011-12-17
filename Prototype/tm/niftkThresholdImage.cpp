/*=========================================================================

 This program takes as an input an  MRI volume and a mask.
 It returns the masked volume.

=========================================================================*/

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionConstIterator.h"
#include "itkImageRegionIterator.h"
#include "itkVector.h"
#include "itkBinaryThresholdImageFilter.h"

#include <iostream>
#include <iomanip>
#include <fstream>

using namespace std;

int main( int argc, char ** argv )
{
   if( argc < 4 ) 
   { 
     std::cerr << "Usage: " << std::endl;
     std::cerr << argv[0] << " inputVolume inputMaskVolume outputBinaryMask(values<threshold) thresholdValue(if not set, then it will create volumes thresholded for each intensity of the inputVolume)";
     std::cerr <<  std::endl;
     return EXIT_FAILURE;
   }
   
   const unsigned int   Dimension = 3;
   typedef float InputPixelType;
   typedef short OutputPixelType;
   
   typedef itk::Image< InputPixelType,  Dimension >   InputImageType;
   typedef itk::Image< OutputPixelType,  Dimension >   MaskImageType;
   
   // reader and writer for the input and output images
   typedef itk::ImageFileReader< InputImageType >  ReaderType;
   typedef itk::ImageFileReader< MaskImageType >  ReaderMaskType;
 
   ReaderType::Pointer reader = ReaderType::New();
   ReaderMaskType::Pointer readerMask = ReaderMaskType::New();
   
   reader->SetFileName( argv[1] );
   readerMask->SetFileName( argv[2] );
   
   // the input and output images
   InputImageType::Pointer inputImage = InputImageType::New(); 
   MaskImageType::Pointer inMaskImage = MaskImageType::New(); 
    
   reader->Update();
   readerMask->Update();
  
   inputImage = reader->GetOutput();
   inMaskImage = readerMask->GetOutput();

   // Minimum and Maximum intensity in the image

   // Iterators declaration
   typedef itk::ImageRegionConstIterator< InputImageType > ConstIteratorType;
   typedef itk::ImageRegionConstIterator< MaskImageType > ConstMaskIteratorType;
  
   ConstIteratorType in( inputImage, inputImage->GetLargestPossibleRegion() );
   ConstMaskIteratorType mask( inMaskImage, inMaskImage->GetLargestPossibleRegion() );
   
   float max = 0;
   float min = 99;
   
   for ( in.GoToBegin(), mask.GoToBegin(); !in.IsAtEnd(); ++in, ++mask )
   {
     if ( mask.Get()>0 )
     {
       if ( in.Get()>max )
         max = in.Get();
       if ( in.Get()<min )
         min = in.Get();
     }
   }
   
   std::cout << "Max: " << max << " min: " << min << std::endl;
 
   typedef itk::BinaryThresholdImageFilter< InputImageType, MaskImageType > BinaryThresholdFilterType;
   BinaryThresholdFilterType::Pointer thresholdFilter = BinaryThresholdFilterType::New();

   thresholdFilter->SetInput( inputImage );   
   thresholdFilter->SetLowerThreshold( min+1 );
   thresholdFilter->SetInsideValue( 1 );
   thresholdFilter->SetOutsideValue( 0 );  

   std::string outName = argv[3];

   typedef itk::ImageFileWriter< MaskImageType  > WriterType;
   WriterType::Pointer writer = WriterType::New();

   // Treshold the image for all values in the range
   //for (int i=(int) 201; i <= (int) 203; i++) 
   for (int i=(int) min+2; i <= (int) max; i++)
   {
     std::cout << "Thresholding volume for value: " << i << std::endl;

     thresholdFilter->SetUpperThreshold(i);

     stringstream volName;
     volName << outName << "-" << i << ".gipl.gz";
     std::cout << "The name is "<< volName.str() << std::endl;
    
     writer->SetFileName( volName.str() );

     thresholdFilter->Update();
     writer->SetInput( thresholdFilter->GetOutput() ); 
     writer->Update();     
     
   }

   return 0;
}


