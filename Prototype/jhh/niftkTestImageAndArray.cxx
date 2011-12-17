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

#include "ConversionUtils.h"
#include "CommandLineParser.h"

#include "itkImageAndArray.h"

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"


struct niftk::CommandLineArgumentDescription clArgList[] = {
  
  {OPT_STRING, "o", "output", "An optional output image file."},

  {OPT_STRING|OPT_LONELY|OPT_REQ, NULL, "input", "An input image file."},

  {OPT_DONE, NULL, NULL, 
   "Program to test the itk::ImageAndArray class"

  }
};


enum {
  O_OUTPUT_IMAGE = 0,

  O_INPUT_IMAGE
};


int main(int argc, char** argv)
{
  std::string fileInputImage;	// Input image filename
  std::string fileOutputImage;	// Output image filename


  typedef float IntensityType;

  typedef itk::ImageAndArray< IntensityType, 3 > ImageType;

  typedef itk::ImageFileReader< ImageType >  InputImageReaderType;
  typedef itk::ImageFileWriter< ImageType >  OutputImageWriterType;

  // Create the command line parser, passing the
  // 'CommandLineArgumentDescription' structure. The final boolean
  // parameter indicates whether the command line options should be
  // printed out as they are parsed.

  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);

  CommandLineOptions.GetArgument(O_INPUT_IMAGE, fileInputImage);
  CommandLineOptions.GetArgument(O_OUTPUT_IMAGE, fileOutputImage);


  // Load the input image
  // ~~~~~~~~~~~~~~~~~~~~

  InputImageReaderType::Pointer inputImageReader  = InputImageReaderType::New();
  
  inputImageReader->SetFileName( fileInputImage );

  try { 
    std::cout << "Reading input 3D volume: " <<  fileInputImage;
    inputImageReader->Update();
    std::cout << "Done";
  } 
  catch( itk::ExceptionObject & err ) { 
    std::cerr << "ERROR: Failed to load input image: " << err << std::endl; 
    return EXIT_FAILURE;
  }                


  // Get an ImageAndArray pointer to the object
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  ImageType::Pointer inputImage = inputImageReader->GetOutput();

  inputImage->SynchronizeArray();

  inputImage->Print(std::cout);


  // Perform some array operations on the image
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  *inputImage += *inputImage;
  
  std::cout << "*inputImage += *inputImage";
  std::cout << *inputImage << std::endl;
  std::cout << "Done";

  *inputImage *= 10;
  
  std::cout << "*inputImage *= 10";
  std::cout << *inputImage << std::endl;
  std::cout << "Done";

  inputImage->Fill(0.);
  
  std::cout << "inputImage->Fill(0.)";
  std::cout << *inputImage << std::endl;
  std::cout << "Done";

  
  // Write the output image
  // ~~~~~~~~~~~~~~~~~~~~~~

  OutputImageWriterType::Pointer writer = OutputImageWriterType::New();

  writer->SetFileName( fileOutputImage );

  writer->SetInput( inputImage );

  try { 
    std::cout << "Writing output to file: " << fileOutputImage;
    writer->Update();
    std::cout << "Done";
  } 
  catch( itk::ExceptionObject & err ) { 
    std::cerr << "ERROR: Failed to write output to file: " << err << std::endl; 
    return EXIT_FAILURE;
  }         


  std::cout << "Done";
}

