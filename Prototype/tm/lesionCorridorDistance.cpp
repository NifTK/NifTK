/*=========================================================================

 This program is used to change the pixel type of an input volume.

=========================================================================*/

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkCastImageFilter.h"
#include "itkVector.h"
#include "itkContinuousIndex.h"
#include "itkImageMomentsCalculator.h"
#include "itkImageRegionConstIterator.h"

#include <iostream>
#include <iomanip>
#include <fstream>

using namespace std;

int main( int argc, char ** argv )
{
   if( argc < 4 ) 
   { 
     std::cerr << "Usage: " << std::endl;
     std::cerr << argv[0] << " lesion2DmaskImage corridor2DmaskImage distance";
     std::cerr <<  std::endl;
     return EXIT_FAILURE;
   }

   const  unsigned int  Dimension = 2;
   typedef float PixelType; 

   typedef itk::Image< PixelType,  Dimension >  ImageType;
   typedef itk::ImageFileReader< ImageType >  ReaderType;

   ReaderType::Pointer reader1 = ReaderType::New();
   ImageType::Pointer image1 = ImageType::New();
   
   ReaderType::Pointer reader2 = ReaderType::New();
   ImageType::Pointer image2 = ImageType::New();
   
   reader1->SetFileName( argv[1] ); 
   reader1->Update();
   image1 = reader1->GetOutput();
   
   reader2->SetFileName( argv[2] ); 
   reader2->Update();
   image2 = reader2->GetOutput();
   
   typedef itk::ImageMomentsCalculator< ImageType >  ImageCalculatorType;
   ImageCalculatorType::Pointer imageCalculator1 = ImageCalculatorType::New();
   imageCalculator1->SetImage( image1 );
   imageCalculator1->Compute();
   ImageCalculatorType::VectorType massCentre1 = imageCalculator1->GetCenterOfGravity();

   // Iterators declaration
   typedef itk::ImageRegionConstIterator< ImageType > ConstIteratorType;
   
   
   ConstIteratorType les( image1, image1->GetLargestPossibleRegion() );
   ConstIteratorType cor( image2, image2->GetLargestPossibleRegion() );
   
   float minDist = 99.;
   float curDist;
   
   for ( les.GoToBegin(), cor.GoToBegin(); !les.IsAtEnd(); ++les, ++cor )
   {
     if ( cor.Get()>0 )
     {
       ImageType::IndexType curIndex = cor.GetIndex();
       ImageType::PointType curPoint;
       image2->TransformIndexToPhysicalPoint( curIndex, curPoint );
       curDist = sqrt( pow(curPoint[0]-massCentre1[0],2) + pow(curPoint[1]-massCentre1[1],2) );
       if ( curDist < minDist )
         minDist = curDist;
     }
   }

   //double distance;
   //distance = sqrt( pow(massCentre2[0]-massCentre1[0],2) + pow(massCentre2[1]-massCentre1[1],2) );

   std::cout << "Mass centre 1: " << massCentre1[0] << " " << massCentre1[1] << std::endl;
   
   std::cout << "Distance is: " << minDist << std::endl;

   ofstream outFile;

   outFile.open( argv[3] );
   
   outFile << "Distance is: " << minDist << std::endl;

   outFile.close();

   return 0;
}
