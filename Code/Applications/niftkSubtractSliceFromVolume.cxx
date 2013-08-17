/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <itkLogHelper.h>
#include <ConversionUtils.h>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkSubtract2DImageFromVolumeSliceFilter.h>

/*!
 * \file niftkSubtractSliceFromVolume.cxx
 * \page niftkSubtractSliceFromVolume
 * \section niftkSubtractSliceFromVolumeSummary Subtracts a 2D image from a specific 3D volume slice.
 */
void Usage(char *exec)
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl
	      << "  Subtracts a 2D image from a specific 3D volume slice." << std::endl << std::endl

	      << "  " << exec 
	      << " -im3D Input3Dimage -im2D Input2Dimage -slice SliceNumber -o Output3Dimage " << std::endl << std::endl

	      << "*** [mandatory] ***" << std::endl << std::endl
	      << "    -im3D  <filename>        Input 3D image volume" << std::endl
	      << "    -im2D  <filename>        Input 2D image" << std::endl
	      << "    -slice <int>             The number of the slice to be subtracted" << std::endl
	      << "    -o     <filename>        Output subtracted 3D volume" << std::endl << std::endl;
  }


/**
 * \brief Project a 3D image volume into 2D.
 */
int main(int argc, char** argv)
{
  std::string fileInputImage3D;
  std::string fileInputImage2D;
  std::string fileOutputImage3D;

  unsigned int iSlice = 0;

  typedef float IntensityType;

  typedef itk::Image< IntensityType, 2> ImageType2D;
  typedef itk::Image< IntensityType, 3> ImageType3D;

  typedef itk::ImageFileReader< ImageType2D > InputImageReaderType2D;
  typedef itk::ImageFileReader< ImageType3D > InputImageReaderType3D;
  typedef itk::ImageFileWriter< ImageType2D > OutputImageWriterType;
  
  typedef itk::Subtract2DImageFromVolumeSliceFilter<IntensityType> Subtract2DImageFromVolumeSliceFilterType;

  // Parse command line args
  // ~~~~~~~~~~~~~~~~~~~~~~~
  

  for(int i=1; i < argc; i++){
    if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 || strcmp(argv[i], "-HELP")==0 
       || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--h")==0){
      Usage(argv[0]);
      return -1;
    }
    else if(strcmp(argv[i], "-im3D") == 0) {
      fileInputImage3D = argv[++i];
      std::cout << "Set -im3D=" << fileInputImage3D << std::endl;
    }
    else if(strcmp(argv[i], "-im2D") == 0) {
      fileInputImage2D = argv[++i];
      std::cout << "Set -im2D=" << fileInputImage2D << std::endl;
    }
    else if(strcmp(argv[i], "-slice") == 0) {
      iSlice = atoi(argv[++i]);
      std::cout << "Set -slice=" << niftk::ConvertToString((int) iSlice) << std::endl;
    }
    else if(strcmp(argv[i], "-o") == 0) {
      fileOutputImage3D = argv[++i];
      std::cout << "Set -o=" << fileOutputImage3D << std::endl;
    }

    else {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return -1;
    }            
  }


  // Validate command line args
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~

  if (fileInputImage3D.length() == 0 || fileInputImage2D.length() == 0 || fileOutputImage3D.length() == 0) {
    Usage(argv[0]);
    return EXIT_FAILURE;
  }


  // Load the input image volume
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~

  InputImageReaderType3D::Pointer inputImageReader3D  = InputImageReaderType3D::New();
  
  inputImageReader3D->SetFileName( fileInputImage3D );

  try { 
    std::cout << "Reading input 3D volume: " <<  fileInputImage3D << std::endl;
    inputImageReader3D->Update();
    std::cout << "Done" << std::endl;
  } 
  catch( itk::ExceptionObject & err ) { 
    std::cerr << "ERROR: Failed to load input image: " << err << std::endl; 
    return EXIT_FAILURE;
  }                
  

  // Load the input image slice
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~

  InputImageReaderType2D::Pointer inputImageReader2D  = InputImageReaderType2D::New();
  
  inputImageReader2D->SetFileName( fileInputImage2D );

  try { 
    std::cout << "Reading input 2D volume: " <<  fileInputImage2D << std::endl;
    inputImageReader2D->Update();
    std::cout << "Done" << std::endl;
  } 
  catch( itk::ExceptionObject & err ) { 
    std::cerr << "ERROR: Failed to load input image: " << err << std::endl; 
    return EXIT_FAILURE;
  }                
  

  // Subtract the required slice
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~

  Subtract2DImageFromVolumeSliceFilterType::Pointer filter = Subtract2DImageFromVolumeSliceFilterType::New();

  filter->SetSliceNumber(iSlice);

  filter->SetInputImage2D(inputImageReader2D->GetOutput());
  filter->SetInputVolume3D(inputImageReader3D->GetOutput());

  filter->Update();


  // Write the output projected image
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  OutputImageWriterType::Pointer writer = OutputImageWriterType::New();

  writer->SetFileName( fileOutputImage3D );
  writer->SetInput( filter->GetOutput() );

  writer->Update();



  std::cout << "Done" << std::endl;
  
  return EXIT_SUCCESS;   
}


