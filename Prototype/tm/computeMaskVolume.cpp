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

#include <iostream>
#include <iomanip>
#include <fstream>

using namespace std;

int main( int argc, char ** argv )
{
   if( argc < 4 ) 
   { 
     std::cerr << "Usage: " << std::endl;
     std::cerr << argv[0] << " inputMaskVolume textFile dimension(either 2 or 3)";
     std::cerr <<  std::endl;
     return EXIT_FAILURE;
   }
   
   int d = atoi(argv[3]);
   
   const unsigned int   Dimension = 3;
   typedef   short  InputPixelType;  //short
   
   typedef itk::Image< InputPixelType,  Dimension >   InputImageType;
   
   // reader and writer for the input and output images
   typedef itk::ImageFileReader< InputImageType >  ReaderType;
   
   ReaderType::Pointer reader = ReaderType::New();
   
   reader->SetFileName( argv[1] );
   
   // the input and output images
   InputImageType::Pointer inputImage = InputImageType::New(); 
   
   reader->Update();
   
   inputImage = reader->GetOutput();
   
   // set the region for the input and the uotput images 
   InputImageType::RegionType inputRegion;
   inputRegion.SetSize( inputImage->GetLargestPossibleRegion().GetSize() );
   InputImageType::RegionType::IndexType inputStart;
   for (unsigned int i=0; i<Dimension; i++)
   {
     inputStart[i] = 0;
   }
   inputRegion.SetIndex( inputStart );
 
   // Iterators declaration
   typedef itk::ImageRegionConstIterator< InputImageType > ConstIteratorType;
   
   ConstIteratorType in( inputImage, inputImage->GetLargestPossibleRegion() );
   
   int numberOfVoxels = 0;
   
   for ( in.GoToBegin(); !in.IsAtEnd(); ++in )
   {
     if ( in.Get()>0 )
       numberOfVoxels++;
   }
   
   std::cout << "Number of non-zero voxels: " << numberOfVoxels << std::endl;
   
   //Get the resolution
   itk::Vector<double, Dimension> spacing = inputImage->GetSpacing();
   
   float volume;
   if (d==3)
     volume = numberOfVoxels*spacing[0]*spacing[1]*spacing[2];
   else if (d==2)
     volume = numberOfVoxels*spacing[0]*spacing[1];
   else
   {
     std::cerr << "Dimension must be either 2 or 3!" << std::endl;
     return 1;
   }
   
   std::cout << "Volume in mm: " << volume << std::endl;
   
   float meanRadius;
   if (d==3)
     meanRadius = pow(volume*0.75/3.14159, 1./3.);
   else if (d==2)
     meanRadius = sqrt(volume/3.14159);
   else
   {
     std::cerr << "Dimension must be either 2 or 3!" << std::endl;
     return 1;
   }
   
   std::cout << "Radius: " << meanRadius << std::endl;
   
   ofstream myFile;
   myFile.open( argv[2] );
  
   myFile << volume << " " << meanRadius << std::endl;

   myFile.close();

   return 0;
}


