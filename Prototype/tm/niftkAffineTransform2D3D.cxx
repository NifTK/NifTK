/*=========================================================================
 
 Program that applies an affine transform to an input volume.

=========================================================================*/

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkAffineTransform2D3D.h"
#include "itkResampleImageFilter.h"
#include "itkLinearInterpolateImageFunction.h"

#include <iostream>
#include <iomanip>
#include <fstream>

using namespace std;

int main( int argc, char * argv[] )
{
 if( argc < 4 ) 
 { 
   std::cerr << "Usage: " << std::endl;
   std::cerr << argv[0] << "  inputImageFile inputParametersFile outputImageFile ";
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
 WriterType::Pointer writer = WriterType::New();
 reader->SetFileName( argv[1] );
 writer->SetFileName( argv[3] );
 
 // definitions of transform and filter
 typedef itk::ResampleImageFilter< InputImageType, OutputImageType > FilterType;
 FilterType::Pointer filter = FilterType::New();

 typedef itk::AffineTransform2D3D< double, dimension > TransformType;
 TransformType::Pointer transform = TransformType::New();
 filter->SetTransform( transform );


 typedef itk::LinearInterpolateImageFunction< InputImageType, double > InterpolatorType;

 InterpolatorType::Pointer interpolator = InterpolatorType::New();
 filter->SetInterpolator( interpolator );
 
 // set the filter options
 filter->SetDefaultPixelValue( 0 ); // value of pixels mapped outside the image
 
 reader->Update(); // update to take the values of reader
 const InputImageType::SpacingType & spacing = reader->GetOutput()->GetSpacing(); 
 const InputImageType::PointType & origin = reader->GetOutput()->GetOrigin();

 InputImageType::SizeType size = reader->GetOutput()->GetLargestPossibleRegion().GetSize();

 std::cout << "Spacing: " << spacing[0] << " "<< spacing[1] << " "<< spacing[2] << std::endl;
 std::cout << "Size: " << size[0] << " "<< size[1] << " "<< size[2] << std::endl; 
 std::cout << "Origin: " << origin[0] << " "<< origin[1] << " "<< origin[2] << std::endl; 

 filter->SetOutputOrigin( origin );
 filter->SetOutputSpacing( spacing );
 filter->SetSize( size );
 
 double center[] = {
   origin[0] + spacing[0]*((double) size[0] - 1.)/2., 
   origin[1] + spacing[1]*((double) size[1] - 1.)/2.,
   origin[2] + spacing[2]*((double) size[2] - 1.)/2.};
 transform->SetCenter( center );

 typedef TransformType::ParametersType ParametersType;
 ParametersType parameters (12); 

 int i = 0; // counter to know the position in the file
 float x; // variable for reading the file params
  
 ifstream inFile;
 inFile.open( argv[2] );
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

 transform->SetParameters( parameters );
 
 filter->SetInput( reader->GetOutput() );
 writer->SetInput( filter->GetOutput() );

 //Parameters of the tranformation matrix used
 std::cout << "Parameters used: " << std::endl;
 std::cout << "Parameters: " << parameters << std::endl;

 transform->Print(std::cout);

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


