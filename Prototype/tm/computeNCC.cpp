/*
 This program computes the similarity measure between the target DRR and
 the DRR that is produced after the final tranform of the source 3D volume.
*/
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkNormalizedCorrelationImageToImageMetric.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkTranslationTransform.h"
#include "itkImageMaskSpatialObject.h"

#include <iostream>
#include <iomanip>
#include <fstream>

using namespace std;

int main( int argc, char * argv[] )
{
 if( argc < 5 ) 
 { 
   std::cerr << "Usage: " << std::endl;
   std::cerr << argv[0] << " target simulatedDRR mask similarity";
   std::cerr <<  std::endl;
   return EXIT_FAILURE;
 }

 // input and output decl  
 const int dimension = 2;
 typedef float PixelType;

 typedef itk::Image< PixelType, dimension > ImageType;
 
 // reader and writer for the input and output images
 typedef itk::ImageFileReader< ImageType >  ReaderType;

 ReaderType::Pointer  fixedImageReader = ReaderType::New();
 ReaderType::Pointer movingImageReader = ReaderType::New();

 fixedImageReader->SetFileName( argv[1] );
 movingImageReader->SetFileName( argv[2] );

 try 
 {
   fixedImageReader->Update();
   movingImageReader->Update();
 }
 catch( itk::ExceptionObject & excep )
 {
   std::cerr << "Exception catched !" << std::endl;
   std::cerr << excep << std::endl;
 }

 typedef itk::NormalizedCorrelationImageToImageMetric< ImageType, ImageType > MetricType;
 MetricType::Pointer metric = MetricType::New();
 
 // Transform and Interpolator are needed from the 'metric'
 typedef itk::TranslationTransform< double, dimension >  TransformType;
 TransformType::Pointer transform = TransformType::New();

 typedef itk::LinearInterpolateImageFunction< ImageType, double >  InterpolatorType;
 InterpolatorType::Pointer interpolator = InterpolatorType::New();

 transform->SetIdentity();
 
 metric->SetTransform( transform );
 metric->SetInterpolator( interpolator );

 metric->SetFixedImage( fixedImageReader->GetOutput() );
 metric->SetMovingImage( movingImageReader->GetOutput() ); 

 //metric->SetFixedImageRegion(  fixedImageReader->GetOutput()->GetBufferedRegion()  );
 typedef itk::ImageMaskSpatialObject< dimension >   MaskType;
 MaskType::Pointer  spatialObjectMask = MaskType::New();
 
 typedef itk::Image< unsigned char, dimension >   ImageMaskType;
 typedef itk::ImageFileReader< ImageMaskType >    MaskReaderType;
 MaskReaderType::Pointer  maskReader = MaskReaderType::New();
 ImageMaskType::Pointer maskImage = ImageMaskType::New();

 maskReader->SetFileName( argv[3] );

 maskReader->Update(); 
 maskImage = maskReader->GetOutput();
 //maskImage->SetOrigin( origin2D );
 spatialObjectMask->SetImage( maskImage );

 metric->SetFixedImageMask( spatialObjectMask );

 //metric->GetFixedImageMask()->Print(std::cout);

 metric->SetFixedImageRegion(fixedImageReader->GetOutput()->GetBufferedRegion());

 fixedImageReader->GetOutput()->Print(std::cout);

 //std::cout << "Before initialising the metric .. " << std::endl;

 try 
 {
   metric->Initialize();
 }
 catch( itk::ExceptionObject & excep )
 {
   std::cerr << "Exception catched !" << std::endl;
   std::cerr << excep << std::endl;
   return EXIT_FAILURE;
 }

 //std::cout << "After initialising the metric, getting the value .. " << std::endl;
 
 std::cout << "Metric value: "<< metric->GetValue( transform->GetParameters() ) << std::endl; 
 
 ofstream myFile;
 myFile.open( argv[4] );

 myFile << metric->GetValue( transform->GetParameters() ) << std::endl;

 myFile.close();

 return 0;

}
