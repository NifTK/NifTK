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
     std::cerr << argv[0] << " vectorImage.mha dx.gipl dy.gipl dz.gipl";
     std::cerr <<  std::endl;
     return EXIT_FAILURE;
   }

   const  unsigned int  Dimension = 3;
   typedef float PixelType;
  
   typedef itk::Image< PixelType,  Dimension > ImageType;
   typedef itk::Vector< PixelType, Dimension > VectorType;

   typedef itk::Image< VectorType, Dimension > DeformationFieldType;

   // reader and writer for the input and output images
   typedef itk::ImageFileReader<  DeformationFieldType  > DeformationReaderType;
   
   DeformationReaderType::Pointer reader = DeformationReaderType::New();
   DeformationFieldType::Pointer field = DeformationFieldType::New();   
   
   typedef itk::ImageFileWriter< ImageType >  WriterType;

   WriterType::Pointer writerX = WriterType::New();
   WriterType::Pointer writerY = WriterType::New();
   WriterType::Pointer writerZ = WriterType::New();

   reader->SetFileName( argv[1] ); 
   writerX->SetFileName( argv[2] );
   writerY->SetFileName( argv[3] );
   writerZ->SetFileName( argv[4] );

   reader->Update();
   field = reader->GetOutput();
   field->DisconnectPipeline();

   std::cout << "Field Origin: " << field->GetOrigin()[0] << " " << field->GetOrigin()[1] << " "<< field->GetOrigin()[2] << std::endl;  
 
   // Creating images
   ImageType::Pointer imageX = ImageType::New();
   imageX->SetOrigin( field->GetOrigin() );
   imageX->SetSpacing( field->GetSpacing() );
   imageX->SetRegions( field->GetLargestPossibleRegion() );
   imageX->SetDirection( field->GetDirection() );
   imageX->Allocate();

   ImageType::Pointer imageY = ImageType::New();
   imageY->SetOrigin( field->GetOrigin() );
   imageY->SetSpacing( field->GetSpacing() );
   imageY->SetRegions( field->GetLargestPossibleRegion() );
   imageY->SetDirection( field->GetDirection() );
   imageY->Allocate();

   ImageType::Pointer imageZ = ImageType::New();
   imageZ->SetOrigin( field->GetOrigin() );
   imageZ->SetSpacing( field->GetSpacing() );
   imageZ->SetRegions( field->GetLargestPossibleRegion() );
   imageZ->SetDirection( field->GetDirection() );
   imageZ->Allocate();

   typedef itk::ImageRegionIteratorWithIndex<ImageType> Iterator;

   Iterator itX( imageX, imageX->GetLargestPossibleRegion() );
   Iterator itY( imageY, imageY->GetLargestPossibleRegion() );
   Iterator itZ( imageZ, imageZ->GetLargestPossibleRegion() );

   VectorType tmpDisplacementVector; 

   for ( itX.Begin(), itY.Begin(), itZ.Begin(); !itX.IsAtEnd(); ++itX,  ++itY,  ++itZ )
   {  
     tmpDisplacementVector = field->GetPixel( itX.GetIndex() );
          
     itX.Set(tmpDisplacementVector[0]);
     itY.Set(tmpDisplacementVector[1]);
     itZ.Set(tmpDisplacementVector[2]);
   }        

   // write the output
   writerX->SetInput( imageX );
   writerY->SetInput( imageY );
   writerZ->SetInput( imageZ );

   try 
   { 
     std::cout << "Writing output image X... " << std::endl;
     writerX->Update();
   } 
   catch( itk::ExceptionObject & err ) 
   { 
     std::cerr << "ERROR: ExceptionObject caught !" << std::endl; 
     std::cerr << err << std::endl; 
   }
   imageX->DisconnectPipeline();

   try 
   { 
     std::cout << "Writing output image Y... " << std::endl;
     writerY->Update();
   } 
   catch( itk::ExceptionObject & err ) 
   { 
     std::cerr << "ERROR: ExceptionObject caught !" << std::endl; 
     std::cerr << err << std::endl; 
   }
   imageY->DisconnectPipeline();

   try 
   { 
     std::cout << "Writing output image Z... " << std::endl;
     writerZ->Update();
   } 
   catch( itk::ExceptionObject & err ) 
   { 
     std::cerr << "ERROR: ExceptionObject caught !" << std::endl; 
     std::cerr << err << std::endl; 
   }
   imageZ->DisconnectPipeline();

   return 0;
}
