/*=========================================================================

 This program is used to convert a vector image *.mha to 3 *.gipl images.

=========================================================================*/

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkVector.h"
#include "itkIndex.h"

int main( int argc, char ** argv )
{
   if( argc < 5 ) 
   { 
     std::cerr << "Usage: " << std::endl;
     std::cerr << argv[0] << " dx.gipl dy.gipl dz.gipl vectorImage.mha";
     std::cerr <<  std::endl;
     return EXIT_FAILURE;
   }

   const  unsigned int  Dimension = 3;
   typedef float PixelType;
  
   typedef itk::Image< PixelType,  Dimension > ImageType;
   typedef itk::Vector< PixelType, Dimension > VectorType;

   typedef itk::Image< VectorType, Dimension > DeformationFieldType;

   // reader and writer for the input and output images
   typedef itk::ImageFileReader<  ImageType  > ImageReaderType;
   
   ImageReaderType::Pointer readerX = ImageReaderType::New();
   ImageReaderType::Pointer readerY = ImageReaderType::New();
   ImageReaderType::Pointer readerZ = ImageReaderType::New();

   ImageType::Pointer dx = ImageType::New();   
   ImageType::Pointer dy = ImageType::New();   
   ImageType::Pointer dz = ImageType::New();   
    
   typedef itk::ImageFileWriter< DeformationFieldType >  WriterType;

   WriterType::Pointer writer = WriterType::New();
   
   readerX->SetFileName( argv[1] ); 
   readerY->SetFileName( argv[2] ); 
   readerZ->SetFileName( argv[3] ); 
   writer->SetFileName( argv[4] );

   readerX->Update();
   dx = readerX->GetOutput();
   dx->DisconnectPipeline();

   readerY->Update();
   dy = readerY->GetOutput();
   dy->DisconnectPipeline();

   readerZ->Update();
   dz = readerZ->GetOutput();
   dz->DisconnectPipeline();

   // Creating output image
   DeformationFieldType::Pointer field = DeformationFieldType::New();
   field->SetOrigin( dx->GetOrigin() );
   field->SetSpacing( dx->GetSpacing() );
   field->SetRegions( dx->GetLargestPossibleRegion() );
   field->SetDirection( dx->GetDirection() );
   field->Allocate();

   typedef itk::ImageRegionIteratorWithIndex< DeformationFieldType > Iterator;

   Iterator it( field, field->GetLargestPossibleRegion() );

   VectorType tmpDisplacementVector; 

   for ( it.Begin(); !it.IsAtEnd(); ++it )
   {  
     tmpDisplacementVector[0] = dx->GetPixel( it.GetIndex() );
     tmpDisplacementVector[1] = dy->GetPixel( it.GetIndex() );
     tmpDisplacementVector[2] = dz->GetPixel( it.GetIndex() );
          
     it.Set( tmpDisplacementVector );
   }        

   // write the output
   writer->SetInput( field );

   try 
   { 
     std::cout << "Writing output vector image... " << std::endl;
     writer->Update();
   } 
   catch( itk::ExceptionObject & err ) 
   { 
     std::cerr << "ERROR: ExceptionObject caught !" << std::endl; 
     std::cerr << err << std::endl; 
   }
   field->DisconnectPipeline();

   return 0;
}
