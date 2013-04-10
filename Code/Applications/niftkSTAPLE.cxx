/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "itkLogHelper.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkSTAPLEImageFilter.h"

/*!
 * \file niftkSTAPLE.cxx
 * \page niftkSTAPLE
 * \section niftkSTAPLESummary Runs ITK STAPLEImageFilter to perform label fusion.
 *
 * STAPLE: The STAPLE algorithm is described in
 * S. Warfield, K. Zou, W. Wells, "Validation of image segmentation and expert quality with an 
 * expectation-maximization algorithm" in MICCAI 2002: Fifth International Conference on 
 * Medical Image Computing and Computer-Assisted Intervention, Springer-Verlag, Heidelberg, Germany, 2002, pp. 298-306.
 * 
 * \li Dimensions: 3
 * \li Pixel type: Scalars only, of unsigned char, char, unsigned short, short. 
 */


/**
 * STAPLE: The STAPLE algorithm is described in
 * S. Warfield, K. Zou, W. Wells, "Validation of image segmentation and expert quality with an 
 * expectation-maximization algorithm" in MICCAI 2002: Fifth International Conference on 
 * Medical Image Computing and Computer-Assisted Intervention, Springer-Verlag, Heidelberg, Germany, 2002, pp. 298-306.
 */
int main(int argc, char** argv)
{
  const unsigned int Dimension = 3;
  typedef short PixelType;
  
  if (argc < 4)
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl;
    std::cout << "  This program perform STAPLE on input segmentations." << std::endl;
    std::cout << " " << std::endl;
    std::cout << " " << argv[0] << " outputFilename foregroundValue confidenceWeight inputFilenames1 inputFilenames2 ..." << std::endl;
    std::cout << " " << std::endl;
    std::cout << "*** [mandatory] ***" << std::endl << std::endl;
    std::cout << "    outputFilename        Output image " << std::endl;
    std::cout << "    foregroundValue       Foreground value in the segmentation. " << std::endl;
    std::cout << "    confidenceWeight      Weight." << std::endl;
    std::cout << "    inputFilenames1       Input segmentations" << std::endl << std::endl;      
    return EXIT_FAILURE;    
  }

  char* outputFilename = argv[1]; 
  PixelType foregroundValue = atoi(argv[2]); 
  double confidenceWeight = atof(argv[3]); 
  std::vector<char*> inputFilenames; 
  for (int argIndex = 4; argIndex < argc; argIndex++)
  {
    inputFilenames.push_back(argv[argIndex]); 
  }

  typedef itk::Image< double, Dimension > OutputImageType;
  typedef itk::Image< PixelType, Dimension > InputImageType;
  typedef itk::ImageFileReader<InputImageType> ImageFileReaderType;

  typedef itk::STAPLEImageFilter<InputImageType, OutputImageType> StapleFilterType;
  StapleFilterType::Pointer stapler = StapleFilterType::New(); 
  stapler->SetConfidenceWeight(confidenceWeight);
  stapler->SetForegroundValue(foregroundValue);

  try
  {
    for (unsigned int inputFileIndex = 0; inputFileIndex < inputFilenames.size(); inputFileIndex++)
    {
      ImageFileReaderType::Pointer reader = ImageFileReaderType::New();
      
      reader->SetFileName(inputFilenames[inputFileIndex]);
      reader->Update();
      stapler->SetInput(inputFileIndex, reader->GetOutput());
    }
  }
  catch (itk::ExceptionObject &e)
  {
    std::cout << "Error while setting input files:" << e << std::endl;
    return EXIT_FAILURE;
  }
  
  try
  {
    itk::ImageFileWriter<OutputImageType>::Pointer writer = itk::ImageFileWriter<OutputImageType>::New();
        
    writer->SetFileName(outputFilename);
    writer->SetInput(stapler->GetOutput());
    writer->Update();
  }
  catch( itk::ExceptionObject &e )
  {
    std::cout << "Error while performing STAPLE:" << e << std::endl;
    return EXIT_FAILURE;
  }
  
  for (unsigned int inputFileIndex = 0; inputFileIndex < inputFilenames.size(); inputFileIndex++)
  {
    std::cout << "Image " << inputFilenames[inputFileIndex] << " : "
      << stapler->GetSpecificity(inputFileIndex) << "," << stapler->GetSensitivity(inputFileIndex) << std::endl;
  }

  return EXIT_SUCCESS; 
}




