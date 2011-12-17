/*=========================================================================

 

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
     std::cerr << argv[0] << " inputMaskVolume inputFatProbs textFile";
     std::cerr <<  std::endl;
     return EXIT_FAILURE;
   }
   
   const unsigned int   Dimension = 3;
   typedef   short  InputPixelType;  //short
   typedef   float  InputProbsPixelType;  //short
   
   typedef itk::Image< InputPixelType,  Dimension >   InputImageType;
   typedef itk::Image< InputProbsPixelType,  Dimension >   InputProbsImageType;
   
   // reader and writer for the input and output images
   typedef itk::ImageFileReader< InputImageType >  ReaderType;
   typedef itk::ImageFileReader< InputProbsImageType >  ProbsReaderType;
   
   ReaderType::Pointer reader = ReaderType::New();
   ProbsReaderType::Pointer fatProbsReader = ProbsReaderType::New();
   
   reader->SetFileName( argv[1] );
   fatProbsReader->SetFileName( argv[2] );
   
   // the input and output images
   InputImageType::Pointer inputImage = InputImageType::New(); 
   InputProbsImageType::Pointer fatProbsImage = InputProbsImageType::New(); 
   
   reader->Update();
   fatProbsReader->Update();
   
   inputImage = reader->GetOutput();
   fatProbsImage = fatProbsReader->GetOutput();
   
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
   typedef itk::ImageRegionConstIterator< InputProbsImageType > ConstProbsIteratorType;
   
   ConstIteratorType in( inputImage, inputImage->GetLargestPossibleRegion() );
   ConstProbsIteratorType inFat( fatProbsImage, fatProbsImage->GetLargestPossibleRegion() );
   
   int numberOfVoxels = 0;
   float fatVoxels = 0;
   
   for ( in.GoToBegin(), inFat.GoToBegin(); !in.IsAtEnd(); ++in, ++inFat )
   {
     if ( in.Get()>0 )
     {
       numberOfVoxels++;
       fatVoxels += inFat.Get();
     }
   }
   
   std::cout << "Number of non-zero voxels: " << numberOfVoxels << std::endl;
   
   //Get the resolution
   itk::Vector<double, Dimension> spacing = inputImage->GetSpacing();
   
   float volume;
   volume = numberOfVoxels*spacing[0]*spacing[1]*spacing[2];
   
   float fatVolume;
   fatVolume = fatVoxels*spacing[0]*spacing[1]*spacing[2];

   std::cout << "Volume in mm^3: " << volume << std::endl;
   std::cout << "Fat volume in mm^3: " << fatVolume << std::endl;
   
   float fatOverVolumeRatio;
   fatOverVolumeRatio = (fatVolume/volume)*100;
   
   std::cout << "Fat over volume ratio: " << fatOverVolumeRatio << std::endl;
   
   ofstream myFile;
   myFile.open( argv[3] );
  
   myFile << volume << " " << fatOverVolumeRatio << std::endl;

   myFile.close();

   return 0;
}


