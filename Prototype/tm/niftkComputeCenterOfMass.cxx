/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-20 14:34:44 +0100 (Tue, 20 Sep 2011) $
 Revision          : $Revision: 7333 $
 Last modified by  : $Author: ad $

 Original author   : j.hipwell@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkCastImageFilter.h"
#include "itkVector.h"
#include "itkContinuousIndex.h"
#include "itkImageMomentsCalculator.h"
#include "itkInvertIntensityImageFilter.h"

#include "CommandLineParser.h"
#include "ConversionUtils.h"

#include <iostream>
#include <iomanip>
#include <fstream>

struct niftk::CommandLineArgumentDescription clArgList[] = {

  {OPT_SWITCH, "inv", NULL, "Invert the input image"},

  {OPT_STRING|OPT_REQ|OPT_LONELY, NULL, "fileVolumeIn", "The input image volume"},
  {OPT_STRING|OPT_REQ|OPT_LONELY, NULL, "fileCofMOut", "The output center of mass text file"},

  {OPT_DONE, NULL, NULL, "Calculate the center of mass of an image."}
};


enum {
  O_INVERT,

  O_INPUT_VOLUME,
  O_OUTPUT_COM
};


using namespace std;

int main( int argc, char ** argv )
{
  bool flgInvert;

  std::string fileInputVolume;
  std::string fileOutputCenterOfMass;

  // Create the command line parser, passing the
  // 'CommandLineArgumentDescription' structure. The final boolean
  // parameter indicates whether the command line options should be
  // printed out as they are parsed.

  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);

  CommandLineOptions.GetArgument(O_INVERT, flgInvert);                 

  CommandLineOptions.GetArgument(O_INPUT_VOLUME, fileInputVolume);          
  CommandLineOptions.GetArgument(O_OUTPUT_COM,   fileOutputCenterOfMass);                 


   const  unsigned int  Dimension = 3;
   typedef float PixelType; 

   typedef itk::Image< PixelType,  Dimension >  ImageType;

   // reader and writer for the input and output images
   typedef itk::ImageFileReader< ImageType >  ReaderType;

   ReaderType::Pointer reader = ReaderType::New();
   ImageType::Pointer image = ImageType::New();

   std::cout << "Reading input volume:" << fileInputVolume;

   reader->SetFileName( fileInputVolume.c_str() ); 

   reader->Update();
   image = reader->GetOutput();

   double origin[] = {0.0, 0.0, 0.0};

   image->SetOrigin( origin );

   typedef itk::ImageMomentsCalculator< ImageType >  ImageCalculatorType;
   ImageCalculatorType::Pointer imageCalculator = ImageCalculatorType::New();

   if ( flgInvert ) {

     typedef itk::MinimumMaximumImageCalculator< ImageType > MinimumMaximumImageCalculatorType;
     MinimumMaximumImageCalculatorType::Pointer maxCalculator = MinimumMaximumImageCalculatorType::New();

     maxCalculator->SetImage( image );
     maxCalculator->Compute();

     typedef itk::InvertIntensityImageFilter< ImageType > InvertFilterType;
     InvertFilterType::Pointer imageInverter = InvertFilterType::New();

     imageInverter->SetInput(image);
     imageInverter->SetMaximum( maxCalculator->GetMaximum() );
     imageInverter->Update();
    
     imageCalculator->SetImage( imageInverter->GetOutput() );
   }
   else
     imageCalculator->SetImage( image );

   std::cout << "Calculating center of mass";

   imageCalculator->Compute();
   ImageCalculatorType::VectorType massCentre = imageCalculator->GetCenterOfGravity();

   std::cout << "Writing coordinate to file: " + fileOutputCenterOfMass;

   ofstream outFile;

   outFile.open( fileOutputCenterOfMass.c_str() );

   for ( unsigned int i=0; i<Dimension; i++ )
   { 
     outFile << massCentre[i] << " ";
   }

   outFile << std::endl;

   outFile.close();

   std::cout << "Center of mass is: "
		   << niftk::ConvertToString( massCentre[0] ) << ", "
		   << niftk::ConvertToString( massCentre[1] ) << ", "
		   << niftk::ConvertToString( massCentre[2] );

   return 0;
}
