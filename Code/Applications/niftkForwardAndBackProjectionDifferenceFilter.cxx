/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "itkLogHelper.h"
#include "ConversionUtils.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkForwardAndBackProjectionDifferenceFilter.h"

/*!
 * \file niftkForwardAndBackProjectionDifferenceFilter.cxx
 * \page niftkForwardAndBackProjectionDifferenceFilter
 * \section niftkForwardAndBackProjectionDifferenceFilterSummar Compute the difference between a set of projection images and a reconstruction estimate (or zero).
 */
void Usage(char *exec)
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl
	      << "  Compute the difference between a set of projection images and a reconstruction estimate (or zero)."
	      << std::endl << std::endl

	      << "  " << exec 
	      << " -projs InputProjectionVolume -o OutputDifference " << std::endl
	      << "  " << std::endl
	      << "*** [mandatory] ***" << std::endl << std::endl
	      << "    -projs <filename>       Input volume of 2D projection images" << std::endl
	      << "    -o    <filename>        Output 3D back-projected image" << std::endl << std::endl
	      << "*** [options]   ***" << std::endl << std::endl
	      << "    -est <filename>         Input current estimate of the 3D volume" << std::endl << std::endl;
  }


/**
 * \brief Project a 3D image volume into 2D.
 */
int main(int argc, char** argv)
{
  std::string fileInputProjectionVolume;
  std::string fileInputCurrentEstimate;
  std::string fileOutputDifference;

  typedef float IntensityType;
  typedef itk::ForwardAndBackProjectionDifferenceFilter< IntensityType > ForwardAndBackProjectionDifferenceFilterType;

  typedef ForwardAndBackProjectionDifferenceFilterType::InputVolumeType InputEstimateType;
  typedef ForwardAndBackProjectionDifferenceFilterType::InputProjectionVolumeType InputProjectionType;  
  typedef ForwardAndBackProjectionDifferenceFilterType::OutputBackProjectedDifferencesType OutputDifferenceType;  

  typedef itk::ImageFileReader< InputEstimateType >    InputEstimateReaderType;
  typedef itk::ImageFileReader< InputProjectionType >  InputProjectionReaderType;

  typedef itk::ImageFileWriter< OutputDifferenceType > OutputImageWriterType;

  // Parse command line args
  // ~~~~~~~~~~~~~~~~~~~~~~~
  

  for(int i=1; i < argc; i++){

    if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 || strcmp(argv[i], "-HELP")==0 
       || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--h")==0){
      Usage(argv[0]);
      return -1;
    }

    else if(strcmp(argv[i], "-projs") == 0) {
      fileInputProjectionVolume = argv[++i];
      std::cout << "Set -projs=" << fileInputProjectionVolume << std::endl;
    }

    else if(strcmp(argv[i], "-est") == 0) {
      fileInputCurrentEstimate = argv[++i];
      std::cout << "Set -est=" << fileInputCurrentEstimate << std::endl;
    }

    else if(strcmp(argv[i], "-o") == 0) {
      fileOutputDifference = argv[++i];
      std::cout << "Set -o=" << fileOutputDifference << std::endl;
    }

    else {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return -1;
    }            
  }


  // Validate command line args
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( fileInputProjectionVolume.length() == 0 || fileOutputDifference.length() == 0 ) {
    Usage(argv[0]);
    return EXIT_FAILURE;
  }
      

  // Create the forward and backward projector
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  ForwardAndBackProjectionDifferenceFilterType::Pointer imReconMetric = ForwardAndBackProjectionDifferenceFilterType::New();

  
  // Load the volume of 2D projection images
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  InputProjectionReaderType::Pointer inputProjectionReader  = InputProjectionReaderType::New();

  inputProjectionReader->SetFileName( fileInputProjectionVolume );

  try { 
    std::cout << "Reading input volume of 2D projection images: " << fileInputProjectionVolume << std::endl;
    inputProjectionReader->Update();
    std::cout << "Done" << std::endl;
  } 
  catch( itk::ExceptionObject & err ) { 
    std::cerr << "ERROR: Failed to load input image: " << err << std::endl; 
    return EXIT_FAILURE;
  }                

  imReconMetric->SetInputProjectionVolume( inputProjectionReader->GetOutput() );


  // Load the current estimate (or create it)
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( fileInputCurrentEstimate.length() != 0 ) {

    InputEstimateReaderType::Pointer inputEstimateReader  = InputEstimateReaderType::New();
  
    inputEstimateReader->SetFileName( fileInputCurrentEstimate );

    try { 
      std::cout << "Reading input 3D estimate: " << fileInputCurrentEstimate << std::endl;
      inputEstimateReader->Update();
      std::cout << "Done" << std::endl;
    } 
    catch( itk::ExceptionObject & err ) { 
      std::cerr << "ERROR: Failed to load input image: " << err << std::endl; 
      return EXIT_FAILURE;
    }         

    imReconMetric->SetInputVolume( inputEstimateReader->GetOutput() );
  }


  // Perform the back projection
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~

  try { 
    imReconMetric->Update();
  } 
  catch( itk::ExceptionObject & err ) { 
    std::cerr << "ERROR: Failed to calculate the reconstruction metric: " << err << std::endl; 
    return EXIT_FAILURE;
  }         


  // Write the output projected inputEstimate
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  OutputImageWriterType::Pointer writer = OutputImageWriterType::New();

  writer->SetFileName( fileOutputDifference );
  writer->SetInput( imReconMetric->GetOutput() );

  try { 
    std::cout << "Writing output to file: " << fileOutputDifference << std::endl;
    writer->Update();
    std::cout << "Done" << std::endl;
  } 
  catch( itk::ExceptionObject & err ) { 
    std::cerr << "ERROR: Failed to write output to file: " << err << std::endl; 
    return EXIT_FAILURE;
  }         

  std::cout << "Done" << std::endl;
  
  return EXIT_SUCCESS;   
}


