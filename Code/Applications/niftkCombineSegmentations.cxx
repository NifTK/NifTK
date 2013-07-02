/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#include <itkLogHelper.h>
#include <ConversionUtils.h>
#include <itkShapeBasedAveragingImageFilter.h>
#include <itkSTAPLEImageFilter.h>
#include <itkUCLLabelVotingImageFilter.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkBinaryThresholdImageFilter.h>
#include <itkThresholdImageFilter.h>

const unsigned int Dimension = 3;
typedef short PixelType;

/*!
 * \file niftkCombineSegmentations.cxx
 * \page niftkCombineSegmentations
 * \section niftkCombineSegmentationsSummary Merges several segmentations together to create a single best segmentation. 
 *
 * This program Merges several segmentations together to create a single best segmentation using 
 *   1. STAPLE: Validation of image segmentation and expert quality with an expectation-maximization algorithm, Warfield et. al, MICCAI, 2002.
 *   2. SBA: Shaped-Based Averaging, Rohlfing and Maurer, TMI, 2007.
 *   3. VOTE: Multi-classifier framework for atlas-based image segmentation. Rohlfing and Maurer, Pattern Recognition Letters, 2005.
 * 
 * \li Dimensions: 3
 * \li Pixel type: Scalars only, of short. 
 */
void StartUsage(char *name)
{
  niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
  std::cout << "  " << std::endl;
  std::cout << "  Merges several segmentations together to create a single best segmentation using " << std::endl;
  std::cout << "    1. STAPLE: Validation of image segmentation and expert quality with an " << std::endl; 
  std::cout << "               expectation-maximization algorithm, Warfield et. al, MICCAI, 2002. " << std::endl; 
  std::cout << "    2. SBA: Shaped-Based Averaging, Rohlfing and Maurer, TMI, 2007." << std::endl; 
  std::cout << "    3. VOTE: Multi-classifier framework for atlas-based image segmentation, " << std::endl; 
  std::cout << "             Rohlfing and Maurer, Pattern Recognition Letters, 2005." << std::endl; 
  std::cout << "  " << std::endl;
  std::cout << "  " << name << " <algo> <algo_params> <foreground> <mrf> <output> <input1> <input2> <input3> ..." << std::endl;
  std::cout << "  " << std::endl;
  std::cout << "*** [mandatory] ***" << std::endl << std::endl;
  std::cout << "  <algo> and <algo_params>   Algorithm and the associated parameters, where <algo> is " << std::endl;
  std::cout << "                               STAPLE for STAPLE," << std::endl;
  std::cout << "                                  <algo_params> for STAPLE is the probability threshold (e.g. 0.5)." << std::endl;
  std::cout << "                                  <foreground> only used in STAPLE because this implementation ONLY WORK FOR binary segmentations." << std::endl;
  std::cout << "                               SBA for Shape based averaging," << std::endl;
  std::cout << "                                  <algo_params> for SBA is the label for undecided voxels." << std::endl;
  std::cout << "                                  Specifiy \"default\" to set it to the largest label+1." << std::endl;
  std::cout << "                                  Specifiy 240 to ask it to pick a random lalbel." << std::endl;
  std::cout << "                               VOTE for voting," << std::endl;
  std::cout << "                                  <algo_params> for VOTE is the label for undecided voxels." << std::endl;
  std::cout << "                                  Specify \"default\" to set it to the largest label+1." << std::endl;
  std::cout << "                                  Specifiy 240 to ask it to pick a random lalbel." << std::endl;
  std::cout << "  <foreground>               Foreground pixel value in the input images, only used by STAPLE" << std::endl; 
  std::cout << "  <output>                   Output file" << std::endl;
  std::cout << "  <input1> <input2> ...      Input files" << std::endl;
  std::cout << std::endl;
  std::cout << "  STAPLE example: " << name << " STAPLE 0.5 1 output.nii input1.nii input2.nii input3.nii ... " << std::endl;
  std::cout << std::endl;
  std::cout << "  SBA example: " << name << " SBA 0 0 output.nii input1.nii input2.nii input3.nii ... " << std::endl;
  std::cout << std::endl;
  std::cout << "  VOTE example: " << name << " VOTE 0 0 output.nii input1.nii input2.nii input3.nii ... " << std::endl;
}

/**
 * Combine segmentation using STAPLE. 
 * \param std::string outputFilename The output filename
 * \param const std::vector<std::string>& inputFilenames The vector storing the input filenames
 * \param PixelType foregroundValue The foreground values in the input images identifying the segmentations. 
 * \param double threshold Threshold probabability value. 
 */
void computeSTAPLE(std::string outputFilename, const std::vector<std::string>& inputFilenames, PixelType foregroundValue, double threshold)
{
  const double confidenceWeightForSTAPLE = 1.0; 
  
  typedef itk::Image< double, Dimension > StapleOutputImageType;
  typedef itk::Image< PixelType, Dimension > InputImageType;
  typedef itk::ImageFileReader<InputImageType> ImageFileReaderType;

  typedef itk::STAPLEImageFilter<InputImageType, StapleOutputImageType> StapleFilterType;
  StapleFilterType::Pointer stapler = StapleFilterType::New(); 
  stapler->SetConfidenceWeight(confidenceWeightForSTAPLE);
  stapler->SetForegroundValue(foregroundValue);
  
  for (unsigned int inputFileIndex = 0; inputFileIndex < inputFilenames.size(); inputFileIndex++)
  {
    ImageFileReaderType::Pointer reader = ImageFileReaderType::New();
    
    reader->SetFileName(inputFilenames[inputFileIndex]);
    reader->Update();
    stapler->SetInput(inputFileIndex, reader->GetOutput());
  }
  
  typedef itk::BinaryThresholdImageFilter<StapleOutputImageType, InputImageType> BinaryThresholdImageFilterType; 
  BinaryThresholdImageFilterType::Pointer binaryThresholdImageFilter = BinaryThresholdImageFilterType::New(); 
  itk::ImageFileWriter<InputImageType>::Pointer writer = itk::ImageFileWriter<InputImageType>::New();
      
  binaryThresholdImageFilter->SetInput(stapler->GetOutput()); 
  binaryThresholdImageFilter->SetLowerThreshold(threshold); 
  binaryThresholdImageFilter->SetUpperThreshold(std::numeric_limits<StapleOutputImageType::PixelType>::max()); 
  binaryThresholdImageFilter->SetInsideValue(foregroundValue); 
  binaryThresholdImageFilter->SetOutsideValue(0); 
  writer->SetFileName(outputFilename);
  writer->SetInput(binaryThresholdImageFilter->GetOutput());
  writer->Update();
  
  for (unsigned int inputFileIndex = 0; inputFileIndex < inputFilenames.size(); inputFileIndex++)
  {
    std::cout << "Image " << inputFilenames[inputFileIndex] << " : "
      << stapler->GetSpecificity(inputFileIndex) << "," << stapler->GetSensitivity(inputFileIndex) << std::endl;
  }
}

/**
 * Combine segmentation using shape based averaging. 
 * \param std::string outputFilename The output filename
 * \param const std::vector<std::string>& inputFilenames The vector storing the input filenames
 * \param PixelType foregroundValue The mean mode used in the SBA.
 * \param std::string userDefinedUndecidedLabel The user-defined undecided label. 
 */
void computeSBA(std::string outputFilename, const std::vector<std::string>& inputFilenames, PixelType foregroundValue, std::string userDefinedUndecidedLabel, double mrf)
{
  typedef itk::Image< PixelType, Dimension > InputImageType;
  typedef itk::ImageFileReader<InputImageType> ImageFileReaderType;

  typedef itk::ShapeBasedAveragingImageFilter<InputImageType, InputImageType> FilterType;
  FilterType::Pointer filter = FilterType::New(); 
  
  if (userDefinedUndecidedLabel != "default")
  {
    PixelType userLabel = static_cast<PixelType>(atof(userDefinedUndecidedLabel.c_str())); 
    
    filter->SetLabelForUndecidedPixels(userLabel); 
    std::cout << "user label=" << userLabel << std::endl;
  }
  
  filter->SetMeanMode(static_cast<FilterType::MeanModeType>(foregroundValue)); 
  for (unsigned int inputFileIndex = 0; inputFileIndex < inputFilenames.size(); inputFileIndex++)
  {
    ImageFileReaderType::Pointer reader = ImageFileReaderType::New();
    
    reader->SetFileName(inputFilenames[inputFileIndex]);
    reader->Update();
    filter->SetInput(inputFileIndex, reader->GetOutput());
  }

  filter->Update();

  itk::ImageFileWriter<InputImageType>::Pointer writer = itk::ImageFileWriter<InputImageType>::New();

  if (mrf <= 0.)
  {
    writer->SetFileName(outputFilename);
    writer->SetInput(filter->GetOutput());
    writer->Update();
  }
  else
  {
#if 0
    itk::ImageFileWriter<FilterType::FloatImageType>::Pointer floatImageWriter = itk::ImageFileWriter<FilterType::FloatImageType>::New();
    floatImageWriter->SetFileName("average_map.img");
    floatImageWriter->SetInput(filter->GetAverageDistanceMap());
    floatImageWriter->Update();
    floatImageWriter->SetFileName("variability_map.img");
    floatImageWriter->SetInput(filter->GetVariabilityMap());
    floatImageWriter->Update();
    floatImageWriter->SetFileName("probability_map.img");
    floatImageWriter->SetInput(filter->GetProbabilityMap());
    floatImageWriter->Update();
    filter->ComputeMRF(filter->GetProbabilityMap(), 0.03, 10);
    floatImageWriter->SetFileName("probability_mrf_map.img");
    floatImageWriter->SetInput(filter->GetProbabilityMap());
    floatImageWriter->Update();
#endif
    filter->ComputeMRF(filter->GetProbabilityMap(), mrf, 10);
    typedef itk::BinaryThresholdImageFilter<FilterType::FloatImageType, InputImageType> BinaryThresholdImageFilterType;
    BinaryThresholdImageFilterType::Pointer thresholdImageFilter = BinaryThresholdImageFilterType::New();
    thresholdImageFilter->SetInput(filter->GetProbabilityMap());
    thresholdImageFilter->SetLowerThreshold(0.);
    thresholdImageFilter->SetUpperThreshold(0.5);
    thresholdImageFilter->SetOutsideValue(255);
    thresholdImageFilter->SetInsideValue(0);

    writer->SetFileName(outputFilename);
    writer->SetInput(thresholdImageFilter->GetOutput());
    writer->Update();
  }
}

/**
 * Combine segmentation using voting. 
 * \param std::string outputFilename The output filename
 * \param const std::vector<std::string>& inputFilenames The vector storing the input filenames
 * \param PixelType foregroundValue The foreground values in the input images identifying the segmentations. 
 * \param std::string userDefinedUndecidedLabel The user-defined undecided label. 
 */
void computeVOTE(std::string outputFilename, const std::vector<std::string>& inputFilenames, PixelType foregroundValue, std::string userDefinedUndecidedLabel)
{
  typedef itk::Image< PixelType, Dimension > InputImageType;
  typedef itk::ImageFileReader<InputImageType> ImageFileReaderType;

  typedef itk::UCLLabelVotingImageFilter<InputImageType, InputImageType> FilterType;
  FilterType::Pointer filter = FilterType::New(); 
  
  if (userDefinedUndecidedLabel != "default")
  {
    PixelType userLabel = static_cast<PixelType>(atof(userDefinedUndecidedLabel.c_str())); 
    
    filter->SetLabelForUndecidedPixels(userLabel); 
    std::cout << "user label=" << userLabel << std::endl;
  }
  
  for (unsigned int inputFileIndex = 0; inputFileIndex < inputFilenames.size(); inputFileIndex++)
  {
    ImageFileReaderType::Pointer reader = ImageFileReaderType::New();
    
    reader->SetFileName(inputFilenames[inputFileIndex]);
    reader->Update();
    filter->SetInput(inputFileIndex, reader->GetOutput());
  }
  itk::ImageFileWriter<InputImageType>::Pointer writer = itk::ImageFileWriter<InputImageType>::New();
      
  writer->SetFileName(outputFilename);
  writer->SetInput(filter->GetOutput());
  writer->Update();
}


int main(int argc, char** argv)
{
  std::string algorithm; 
  std::string algorithmParameters; 
  PixelType foregroundValue; 
  std::string outputFilename; 
  std::vector<std::string> inputFilenames; 
  double mrf = 0.;

  if (argc < 7)
  {
    StartUsage(argv[0]); 
    return EXIT_FAILURE; 
  }

  

  int argIndex = 1; 
  algorithm = argv[argIndex];
  std::cout << "algo=" << algorithm<< std::endl;
  argIndex += 1; 
  algorithmParameters = argv[argIndex]; 
  std::cout << "algo_params=" << algorithmParameters<< std::endl;
  argIndex += 1; 
  foregroundValue = atoi(argv[argIndex]); 
  std::cout << "foreground=" << niftk::ConvertToString(foregroundValue)<< std::endl;
  argIndex += 1; 
  mrf = atof(argv[argIndex]);
  std::cout << "mrf=" << mrf << std::endl;
  argIndex += 1;
  outputFilename = argv[argIndex];
  std::cout << "output=" << outputFilename<< std::endl;
  argIndex += 1; 
  for (; argIndex < argc; argIndex++)
  {
    inputFilenames.push_back(argv[argIndex]);  
    std::cout << "input=" << inputFilenames[inputFilenames.size()-1]<< std::endl;
  }

  try
  {
    if (algorithm == "STAPLE")
    {
      computeSTAPLE(outputFilename, inputFilenames, foregroundValue, atof(algorithmParameters.c_str())); 
    }
    else if (algorithm == "SBA")
    {
      computeSBA(outputFilename, inputFilenames, foregroundValue, algorithmParameters, mrf);
    }
    else if (algorithm == "VOTE")
    {
      computeVOTE(outputFilename, inputFilenames, foregroundValue, algorithmParameters); 
    }
    else
    {
      StartUsage(argv[0]); 
      return EXIT_FAILURE; 
    }
  }
  catch (itk::ExceptionObject &e)
  {
    std::cout << "Exception caught:" << e << std::endl;
    return EXIT_FAILURE;
  }
    
  return EXIT_SUCCESS; 
}
    
    
    
  
  
