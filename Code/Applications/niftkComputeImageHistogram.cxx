/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <math.h>
#include <float.h>
#include <iomanip>

#include <ConversionUtils.h>
#include <CommandLineParser.h>

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkScalarImageToHistogramGenerator.h>


struct niftk::CommandLineArgumentDescription clArgList[] = {

  {OPT_SWITCH, "v", NULL, "Verbose output."},

  {OPT_INT, "nbins", "n", "The number of bins to use [64]."},
  {OPT_DOUBLE, "ms", "scale", "The marginal scale of the histogram [10.]."},

  {OPT_STRING, "o", "filename", "The output histogram."},

  {OPT_STRING|OPT_LONELY|OPT_REQ, NULL, "filename", "The input image."},
  
  {OPT_DONE, NULL, NULL, 
   "Program to calculate the histogram of an image.\n"
  }
};


enum {
  O_VERBOSE,

  O_NUMBER_OF_BINS,
  O_MARGINAL_SCALE,
  
  O_OUTPUT_FILE,

  O_INPUT_IMAGE
};


int main( int argc, char *argv[] )
{
  bool flgVerbose = 0;
  int nbins = 64;

  double marginalScale = 0.;

  std::string fileOutput;
  std::string fileInputImage;


  typedef float InputPixelType;
  typedef itk::Image<InputPixelType, 3> InputImageType;

  typedef itk::ImageFileReader< InputImageType > FileReaderType;

  FileReaderType::Pointer imageReader = FileReaderType::New();

  typedef itk::Statistics::ScalarImageToHistogramGenerator< InputImageType > HistogramGeneratorType;

  typedef HistogramGeneratorType::HistogramType  HistogramType;


  // Create the command line parser, passing the
  // 'CommandLineArgumentDescription' structure. The final boolean
  // parameter indicates whether the command line options should be
  // printed out as they are parsed.

  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);

  CommandLineOptions.GetArgument( O_VERBOSE, flgVerbose );

  CommandLineOptions.GetArgument( O_NUMBER_OF_BINS, nbins );
  CommandLineOptions.GetArgument( O_MARGINAL_SCALE, marginalScale );

  CommandLineOptions.GetArgument( O_OUTPUT_FILE, fileOutput );

  CommandLineOptions.GetArgument( O_INPUT_IMAGE, fileInputImage );


  // Read the input image
  // ~~~~~~~~~~~~~~~~~~~~

  imageReader->SetFileName( fileInputImage.c_str() );

  try
  { 
    std::cout << "Reading image: " << fileInputImage << std::endl;
    imageReader->Update();
  }
  catch (itk::ExceptionObject &ex)
  { 
    std::cout << ex << std::endl;
    return EXIT_FAILURE;
  }


  // Calculate the histogram
  // ~~~~~~~~~~~~~~~~~~~~~~~

  HistogramGeneratorType::Pointer histogramGenerator = HistogramGeneratorType::New();

  histogramGenerator->SetNumberOfBins( nbins );

  if ( marginalScale )
    histogramGenerator->SetMarginalScale( marginalScale );
  else
    histogramGenerator->SetMarginalScale( 10. );

  histogramGenerator->SetInput(  imageReader->GetOutput() );
  histogramGenerator->Compute();


  const HistogramType *histogram = histogramGenerator->GetOutput();
    
  HistogramType::ConstIterator itr = histogram->Begin();
  HistogramType::ConstIterator end = histogram->End();

  unsigned int binNumber = 0;

  if ( flgVerbose ) 
  {
    while( itr != end )
    {
      std::cout << "bin number = " << std::setw(12) << binNumber 
		<< " bin value = " << std::setw(12) << histogram->GetMeasurement(binNumber, 0) 
		<< " frequency = " << std::setw(12) << itr.GetFrequency() << std::endl;     
      
      ++itr;
      ++binNumber;
    }    
  }


  // Write the histogram to a text file
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if (fileOutput.length() != 0) {

    std::fstream fout;

    fout.open(fileOutput.c_str(), std::ios::out);
    
    if ((! fout) || fout.bad()) {
      std::cerr << "ERROR: Failed to open file: " << fileOutput.c_str() << std::endl;
      return EXIT_FAILURE;
    }

    std::cout << "Writing histogram to file: " << fileOutput.c_str() << std::endl;

    itr = histogram->Begin();
    end = histogram->End();

    binNumber = 0;

    while( itr != end )
    {
      fout << histogram->GetMeasurement(binNumber, 0) << " " 
	   << itr.GetFrequency() << std::endl;     

      ++itr;
      ++binNumber;
    }    

    fout.close();
  }
  

  return EXIT_SUCCESS;
}
