/*=========================================================================
 
 Program that applies an affine transform to an input volume and projects it
 in 2D.

=========================================================================*/

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkNewAffineTransform.h"
#include "itkResampleImageFilter.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkRayCastInterpolateImageFunction.h"
#include "itkImageMomentsCalculator.h"

#include <iostream>
#include <iomanip>
#include <fstream>

using namespace std;

int main( int argc, char * argv[] )
{
 if( argc < 5 ) 
 { 
   std::cerr << "Usage: " << std::endl;
   std::cerr << argv[0] << " input3DImageFile input3DMaskImageFile input2DimageFile inputParametersFile output2DImageFile ";
   std::cerr <<  std::endl;
   return EXIT_FAILURE;
 }
 // input and output decl  
 const int dimension = 3;
 typedef  float   InputPixelType;
 typedef  float   OutputPixelType;
  
 typedef itk::Image< InputPixelType,  dimension >   InputImageType;
 typedef itk::Image< OutputPixelType, dimension  >   OutputImageType; 
 
 // reader and writer for the input and output images
 typedef itk::ImageFileReader< InputImageType >  ReaderType;
 typedef itk::ImageFileWriter< OutputImageType >  WriterType;

 ReaderType::Pointer reader = ReaderType::New();
 ReaderType::Pointer readerMask = ReaderType::New();
 ReaderType::Pointer reader2D = ReaderType::New();
 WriterType::Pointer writer = WriterType::New();
 reader->SetFileName( argv[1] );
 readerMask->SetFileName( argv[2] );
 reader2D->SetFileName( argv[3] );
 writer->SetFileName( argv[5] );
 
 // definitions of transform and filter
 typedef itk::ResampleImageFilter< InputImageType, OutputImageType > FilterType;
 FilterType::Pointer filter = FilterType::New();

 typedef itk::NewAffineTransform< double, dimension > TransformType;
 TransformType::Pointer transform = TransformType::New();
 filter->SetTransform( transform );

 reader->Update(); // update to take the values of reader
 const InputImageType::SpacingType & spacing3D = reader->GetOutput()->GetSpacing(); 
 //const InputImageType::PointType & origin3D = reader->GetOutput()->GetOrigin();
 InputImageType::SizeType size3D = reader->GetOutput()->GetLargestPossibleRegion().GetSize();

 // interpolator type to evaluate intensities at non-grid positions
 typedef itk::RayCastInterpolateImageFunction< InputImageType, double > InterpolatorType;
 InterpolatorType::Pointer interpolator = InterpolatorType::New();

 //double center[] = {
 //  origin3D[0] + spacing3D[0]*((double) size3D[0] - 1.)/2., 
 //  origin3D[1] + spacing3D[1]*((double) size3D[1] - 1.)/2.,
 // origin3D[2] + spacing3D[2]*((double) size3D[2] - 1.)/2.};
 typedef itk::ImageMomentsCalculator< InputImageType >  ImageCalculatorType;
 ImageCalculatorType::Pointer imageCalculator = ImageCalculatorType::New();
 reader->Update();
 imageCalculator->SetImage(reader->GetOutput());
 imageCalculator->Compute();
 ImageCalculatorType::VectorType massCentre = imageCalculator->GetCenterOfGravity();

 std::cout<<"Mass centre of the image: "<<massCentre[0]<<" "<<massCentre[1]<<" "<<massCentre[2]<<std::endl;
 
 double center[3];
 
 center[0] = massCentre[0];
 center[1] = massCentre[1];
 center[2] = massCentre[2];

 transform->SetCenter( center );

 // set the origin for the 2D image
 double origin2D[ dimension ];

 reader2D->Update(); // update to take the values of reader
 const InputImageType::SpacingType & resolution2D = reader2D->GetOutput()->GetSpacing(); 
 InputImageType::SizeType size2D = reader2D->GetOutput()->GetLargestPossibleRegion().GetSize();
 
 ImageCalculatorType::Pointer Image2DCalculator = ImageCalculatorType::New();
 reader2D->Update();
 Image2DCalculator->SetImage(reader2D->GetOutput());
 Image2DCalculator->Compute();
 ImageCalculatorType::VectorType massCentre2D = Image2DCalculator->GetCenterOfGravity();

 std::cout<<"Mass centre of the 2D image: "<<massCentre2D[0]<<" "<<massCentre2D[1]<<std::endl;

 origin2D[0] = center[0] - massCentre2D[0]; 
 origin2D[1] = center[1] - massCentre2D[1]; 
 origin2D[2] = center[2] + 90.;

 //origin2D[0] = center[0] - resolution2D[0]*((double) size2D[0] - 1.)/2.; 
 //origin2D[1] = center[1] - resolution2D[1]*((double) size2D[1] - 1.)/2.; 
 //origin2D[2] = center[2] + 90. ; //160 
 
 //std::cout<<"origin2D: "<<origin2D[0]<<" "<<origin2D[1]<<" " <<origin2D[2]<<std::endl;

 // Initialisation of the interpolator
 float sid = 660.;
 InterpolatorType::InputPointType focalpoint;

 focalpoint[0]= center[0];
 focalpoint[1]= center[1];
 focalpoint[2]= center[2] - (sid-90.); // 160// 

 interpolator->SetFocalPoint(focalpoint);

 //std::cout<<"focalpoint: "<<focalpoint[0]<<" "<<focalpoint[1]<<" " <<focalpoint[2]<<std::endl;


 filter->SetInterpolator( interpolator );
 
 // set the filter options
 filter->SetDefaultPixelValue( 0 ); // value of pixels mapped outside the image
 
 filter->SetOutputOrigin( origin2D );
 filter->SetOutputSpacing( resolution2D );
 filter->SetSize( size2D );
 
 typedef TransformType::ParametersType ParametersType;
 ParametersType parameters (12); 

 int i = 0; // counter to know the position in the file
 float x; // variable for reading the file params

 //std::cout << "Reading the parameters.." << std::endl;
  
 ifstream inFile;
 inFile.open( argv[4] );
 if (!inFile) 
 {
   std::cout << "Unable to open file";
   exit(1); // terminate with error
 }
 while (inFile >> x) 
 {
   if ( i < 12 )
   {
     parameters[i] = x;
     i++;
   }
   else
     break;
 }

 //std::cout << "The parameters are read" << std::endl;
 double halfDim3D[ dimension ];

 halfDim3D[0] = spacing3D[0]*((double) size3D[0])/2.; 
 halfDim3D[1] = spacing3D[1]*((double) size3D[1])/2.; 
 halfDim3D[2] = spacing3D[2]*((double) size3D[2])/2.; 

 // correct the translation
 // we do not want the volume to be translated and centered to [0, 0, 0] 
 parameters[9] -= halfDim3D[0];
 parameters[10] -= halfDim3D[1];
 parameters[11] -= halfDim3D[2];

 transform->SetParameters( parameters );

 //transform->Print( std::cout );

 interpolator->SetTransform(transform); 

 readerMask->Update();
 filter->SetInput( readerMask->GetOutput() );

 filter->Update();
 OutputImageType::Pointer outputImage = OutputImageType::New();
 outputImage = filter->GetOutput();

 double myOrigin[] = {0, 0, 0}; // used to reset the origin of the DRR

 outputImage->Update();
 outputImage->SetOrigin( myOrigin ); 
 
 writer->SetInput( outputImage );

 //Parameters of the tranformation matrix used
 std::cout << "Parameters used: " << std::endl;
 std::cout << "Parameters: " << parameters << std::endl;

 //transform->Print(std::cout);

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


