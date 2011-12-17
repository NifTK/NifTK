#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#ifdef __BORLANDC__
#define ITK_LEAN_AND_MEAN
#endif


#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
//#include "itkCenteredRigid2DTransform.h"
#include "itkAffineTransform.h"
#include "itkResampleImageFilter.h"
#include "itkLinearInterpolateImageFunction.h"


int main( int argc, char * argv[] )
{
 if( argc < 3 ) 
 { 
   std::cerr << "Usage: " << std::endl;
   std::cerr << argv[0] << "  inputImageFile  outputImageFile ";
   std::cerr <<  std::endl;
   return EXIT_FAILURE;
 }
 // imput and output decl  
 const int dimension = 3;
 typedef    float    InputPixelType; 
 typedef    float    OutputPixelType;
  
 typedef itk::Image< InputPixelType,  dimension >   InputImageType;
 typedef itk::Image< OutputPixelType, dimension  >   OutputImageType; 
 
 // reader and writer for the input and output images
 typedef itk::ImageFileReader< InputImageType >  ReaderType;
 typedef itk::ImageFileWriter< OutputImageType >  WriterType;

 ReaderType::Pointer reader = ReaderType::New();
 WriterType::Pointer writer = WriterType::New();
 reader->SetFileName( argv[1] );
 writer->SetFileName( argv[2] );
 
 // defs of transorm and filter
 typedef itk::ResampleImageFilter< InputImageType, OutputImageType > FilterType;
 FilterType::Pointer filter = FilterType::New();

 typedef itk::AffineTransform< double, dimension > TransformType;
 TransformType::Pointer transform = TransformType::New();

 typedef itk::LinearInterpolateImageFunction< InputImageType, double > InterpolatorType;

 InterpolatorType::Pointer interpolator = InterpolatorType::New();
 filter->SetInterpolator( interpolator );
 
 // set the filter options
 filter->SetDefaultPixelValue( 0 ); // value of pixles mapped outside the image
 
 reader->Update(); // update to take the values of reader
 const InputImageType::SpacingType & spacing = reader->GetOutput()->GetSpacing(); 
 const InputImageType::PointType & origin = reader->GetOutput()->GetOrigin();

 InputImageType::SizeType size = reader->GetOutput()->GetLargestPossibleRegion().GetSize();

 std::cerr << "Spacing: " << spacing[0] << " "<< spacing[1] << " "<< spacing[2] << std::endl;
 std::cerr << "Size: " << size[0] << " "<< size[1] << " "<< size[2] << std::endl;
 std::cerr << "Origin: " << origin[0] << " "<< origin[1] << " "<< origin[2] << std::endl;
 
 filter->SetOutputSpacing( spacing );
 filter->SetOutputOrigin( origin );

 // what is the origin?
 std::cerr << "Origin: " << origin[0] << " "<< origin[1] << " "<< origin[2] << std::endl;

 filter->SetSize( size );

 // set the new size
 InputImageType::SizeType newSize;
 newSize[0] = 352;//229;  //352;//229;//501;
 newSize[1] = 280;//191;  //280;//191;//501;
 newSize[2] = 1;
 filter->SetSize( newSize );
 std::cerr << "New Size: " << newSize[0] << " "<< newSize[1] << " "<< newSize[2] << std::endl;
 
 // transform build
 TransformType::OutputVectorType translation;
  
 translation[0] = ( ( (size[0]-1)/2 ) - ( ((float)newSize[0]-1)/2 ) )*spacing[0];
 translation[1] = ( ( (size[1]-1)/2 ) - ( (float)newSize[1] )  )*spacing[1];
 //translation[0] = -( ( ((float)newSize[0]-1)/2 ) - ( (size[0]-1)/2 ) )*spacing[0];
 //translation[1] = -( ( ((float)newSize[1]-1)/2 ) - ( size[1] )  )*spacing[1];
 translation[2] = 0;
 transform->Translate( translation );

 filter->SetTransform( transform );
 

 filter->SetInput( reader->GetOutput() );
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


