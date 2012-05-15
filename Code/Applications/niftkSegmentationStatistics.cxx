/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-24 17:44:42 +0000 (Thu, 24 Nov 2011) $
 Revision          : $Revision: 7864 $
 Last modified by  : $Author: kkl $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#include "ConversionUtils.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkImageRegionConstIterator.h"
#include "itkLogHelper.h"

/*!
 * \file niftkSegmentationStatistics.cxx
 * \page niftkSegmentationStatistics
 * \section niftkSegmentationStatisticsSummary Computes segmentation statistics between different segmentations. Initially based on Shattuck et. al. NeuroImage 45(2009) 431-439.
 */
void Usage(char *name)
{
  niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
  std::cout << "  " << std::endl;
  std::cout << "  Computes segmentation statistics between different segmentations. Initially based on Shattuck et. al. NeuroImage 45(2009) 431-439." << std::endl;
  std::cout << "  " << std::endl;
  std::cout << "  " << name << " -si groundTruthSegmentedImage [options] segmentedExample1 segmentedExample2 ... segmentedExampleN" << std::endl;
  std::cout << "  " << std::endl;  
  std::cout << "*** [mandatory] ***" << std::endl << std::endl;
  std::cout << "    -si <filename>                 Ground truth segmented image " << std::endl;
  std::cout << "                                   followed by at least one other image " << std::endl << std::endl;
  std::cout << "*** [options]   ***" << std::endl << std::endl;   
  std::cout << "    -lt <int> [1]                  Lower threshold " << std::endl;
  std::cout << "    -ut <int> [max]                Upper threshold " << std::endl;
  std::cout << "    -noname                        Don't output image name " << std::endl;
  std::cout << "    -debug                         Turn on debugging" << std::endl;
  std::cout << "    -fp <file>                     Output a binary image with the false positve voxels" << std::endl;
  std::cout << "    -fn <file>                     Output a binary image with the false negative voxels" << std::endl;
  std::cout << "    -ignore 0/1/2 g/s slice        Ignore slices (g)reater or (s)maller the specified nubmer" << std::endl; 
}

/**
 * \brief Computes segmentation stats (like sensitivity, specificity etc).
 */
int main(int argc, char** argv)
{
  const unsigned int Dimension = 3;
  typedef        int PixelType;
  
  std::string exampleSegmentedImageFileName = "";
  std::string groundTruthSegmentedImageFileName = ""; 
  
  int lower = std::numeric_limits<PixelType>::min();
  int higher = std::numeric_limits<PixelType>::min();
  int firstArgumentAfterOptions = std::numeric_limits<PixelType>::min();
  bool outputImageName=true;
  bool debug = false;
  int ignoreAxis = 0; 
  char ignoreDirection = 'g'; 
  int ignoreSlice = -1; 
  std::string outputFpFilename; 
  std::string outputFnFilename; 
  

  for(int i=1; i < argc; i++){
    if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 || strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--h")==0){
      Usage(argv[0]);
      return EXIT_SUCCESS;
    }
    else if(strcmp(argv[i], "-debug") == 0){
      std::cout << "Set -debug=" << niftk::ConvertToString(debug) << std::endl;
    }            
    else if(strcmp(argv[i], "-si") == 0){
      groundTruthSegmentedImageFileName=argv[++i];
      std::cout << "Set -si=" << groundTruthSegmentedImageFileName << std::endl;
    }
    else if(strcmp(argv[i], "-lt") == 0){
      lower=atoi(argv[++i]);
      std::cout << "Set -lt=" << niftk::ConvertToString(lower) << std::endl;
    }
    else if(strcmp(argv[i], "-ut") == 0){
      higher=atoi(argv[++i]);
      std::cout << "Set -ut=" << niftk::ConvertToString(higher) << std::endl;
    }
    else if(strcmp(argv[i], "-noname") == 0){
      outputImageName=false;
      std::cout << "Set -noname=" << niftk::ConvertToString(outputImageName) << std::endl;
    }
    else if(strcmp(argv[i], "-fp") == 0){
      outputFpFilename = argv[++i];
      std::cout << "Set -fp=" << outputFpFilename << std::endl;
    }
    else if(strcmp(argv[i], "-fn") == 0){
      outputFnFilename = argv[++i];
      std::cout << "Set -fn=" << outputFnFilename << std::endl;
    }
    else if (strcmp(argv[i], "-ignore") == 0){
      ignoreAxis = atoi(argv[++i]); 
      ignoreDirection = *(argv[++i]); 
      ignoreSlice = atoi(argv[++i]); 
      if (ignoreDirection != 'g' || ignoreDirection != 's')
      {
        std::cout << "Direction must be either g or s." << std::endl; 
        return EXIT_FAILURE; 
      }
      std::cout << "Set -ignore=" << ignoreAxis << "," << ignoreDirection << "," << ignoreSlice << std::endl;
    }
    else 
      {
        if(firstArgumentAfterOptions < 0)
          {
            firstArgumentAfterOptions = i;  
            std::cout << "First agument, assumed to be an image=" << niftk::ConvertToString((int)firstArgumentAfterOptions) << std::endl;
          }
      }
  }
  
  if (argc == 1)
    {
      Usage(argv[0]);
      return EXIT_FAILURE;
    }
  
  if (argc < 3)
    {
      std::cerr << argv[0] << ":\tYou should specify at least 2 images" << std::endl;
      return EXIT_FAILURE;
    }

  if (groundTruthSegmentedImageFileName.length() == 0)
    {
      std::cerr << argv[0] << ":\tYou didn't specify a ground truth image" << std::endl;
      return EXIT_FAILURE;
    }
  
  if (firstArgumentAfterOptions < 0)
    {
      std::cerr << argv[0] << ":\tYou didn't specify an image to test" << std::endl;
      return EXIT_FAILURE;
    }

  typedef itk::Image< PixelType, Dimension > InputImageType;
  typedef itk::ImageFileReader< InputImageType >  ImageReaderType;
  typedef itk::ImageFileWriter< InputImageType >  ImageWriterType;
  typedef itk::BinaryThresholdImageFilter<InputImageType, InputImageType> BinaryThresholdFilterType;
  
  ImageReaderType::Pointer groundTruthReader = ImageReaderType::New();
  ImageReaderType::Pointer segmentedImageReader = ImageReaderType::New();
  BinaryThresholdFilterType::Pointer groundTruthThresholder = BinaryThresholdFilterType::New();
  BinaryThresholdFilterType::Pointer segmentedImageThresholder = BinaryThresholdFilterType::New();
  ImageReaderType::Pointer falsePositiveImageReader;
  ImageReaderType::Pointer falseNegativeImageReader;
  itk::ImageRegionIterator<InputImageType>* falsePositiveImageIterator = NULL;
  itk::ImageRegionIterator<InputImageType>* falseNegativeImageIterator = NULL;
  
  // Load in the ground truth image to get all the meta-data the same as the ground truth image. 
  // Set up the iterators and initialise the images to be 0.
  if (outputFpFilename.length() > 0)
  {
    falsePositiveImageReader = ImageReaderType::New();
    falsePositiveImageReader->SetFileName(groundTruthSegmentedImageFileName); 
    falsePositiveImageReader->Update(); 
    
    falsePositiveImageIterator = new itk::ImageRegionIterator<InputImageType>(falsePositiveImageReader->GetOutput(), falsePositiveImageReader->GetOutput()->GetLargestPossibleRegion());
    for (falsePositiveImageIterator->GoToBegin(); !falsePositiveImageIterator->IsAtEnd(); ++(*falsePositiveImageIterator))
      falsePositiveImageIterator->Set(0); 
  }
  if (outputFnFilename.length() > 0)
  {
    falseNegativeImageReader = ImageReaderType::New();
    falseNegativeImageReader->SetFileName(groundTruthSegmentedImageFileName); 
    falseNegativeImageReader->Update(); 
    
    falseNegativeImageIterator = new itk::ImageRegionIterator<InputImageType>(falseNegativeImageReader->GetOutput(), falseNegativeImageReader->GetOutput()->GetLargestPossibleRegion());
    for (falseNegativeImageIterator->GoToBegin(); !falseNegativeImageIterator->IsAtEnd(); ++(*falseNegativeImageIterator))
      falseNegativeImageIterator->Set(0); 
  }

  // Set the thresholds.
  if (lower == std::numeric_limits<PixelType>::min())
    {
      lower = 1;
    }
  if (higher == std::numeric_limits<PixelType>::min())
    {
      higher =  std::numeric_limits<PixelType>::max(); 
    }

  std::cout << "Using lower thresold " << niftk::ConvertToString(lower) << ", and higher threshold " << niftk::ConvertToString(higher) << std::endl;

  try 
    {
      std::cout << "Loading:" << groundTruthSegmentedImageFileName << std::endl;
      
      groundTruthReader->SetFileName(groundTruthSegmentedImageFileName);
      groundTruthReader->Update();
      
      std::cout << "Done" << std::endl;
      
    }     
  catch (itk::ExceptionObject& exceptionObject)
    {
      std::cout << "Failed to load ground truth image " << groundTruthSegmentedImageFileName << ", with exception " << exceptionObject << std::endl;
      return EXIT_FAILURE; 
    }

  groundTruthThresholder->SetInput(groundTruthReader->GetOutput());
  groundTruthThresholder->SetLowerThreshold(lower);
  groundTruthThresholder->SetUpperThreshold(higher);
  groundTruthThresholder->SetInsideValue(1);
  groundTruthThresholder->SetOutsideValue(0);
  groundTruthThresholder->UpdateLargestPossibleRegion();
  
  std::cout << "Thresholded:" << groundTruthSegmentedImageFileName << std::endl;

  if (outputImageName)
    {
      std::cout << "Image name\t";  
    }
  std::cout << "True Positive\tTrue Negative\tFalse Positive\tFalse Negative\tSensitivity\tSpecificity\tFalse Positive Rate\tFalse Negative Rate\tJaccard\tDice\tConformity\tSensibility" << std::endl;
  
  // Now loop around the remaining arguments calculating the numbers
  unsigned int i = 0;
  
  for (i = firstArgumentAfterOptions; i < (unsigned int)argc; i++)
    {
      exampleSegmentedImageFileName = argv[i];
      
      try
        {
          std::cout << "Loading:" << exampleSegmentedImageFileName << std::endl;
          segmentedImageReader->SetFileName(exampleSegmentedImageFileName);  
          segmentedImageReader->Update();
          std::cout << "Done" << std::endl;
        }
      catch (itk::ExceptionObject& exceptionObject)
        {
          std::cout << "Failed to load segmented image " << exampleSegmentedImageFileName << ", due to exception " << exceptionObject << ", so stopping" << std::endl;
          return EXIT_FAILURE;
        }
      
      if (segmentedImageReader->GetOutput()->GetLargestPossibleRegion().GetSize() != 
        groundTruthReader->GetOutput()->GetLargestPossibleRegion().GetSize())
        {
          std::cout << "Images should have the same number of voxels in all dimensions" << std::endl;
          return EXIT_FAILURE;
        }
      
      segmentedImageThresholder->SetInput(segmentedImageReader->GetOutput());
      segmentedImageThresholder->SetLowerThreshold(lower);
      segmentedImageThresholder->SetUpperThreshold(higher);
      segmentedImageThresholder->SetInsideValue(1);
      segmentedImageThresholder->SetOutsideValue(0);
      segmentedImageThresholder->UpdateLargestPossibleRegion();
      
      itk::ImageRegionConstIterator<InputImageType> groundTruthIterator(groundTruthThresholder->GetOutput(), groundTruthThresholder->GetOutput()->GetLargestPossibleRegion());
      itk::ImageRegionConstIteratorWithIndex<InputImageType> segmentedImageIterator(segmentedImageThresholder->GetOutput(), segmentedImageThresholder->GetOutput()->GetLargestPossibleRegion());

      PixelType groundTruthValue;
      PixelType segmentedImageValue;
      
      double sensitivity = 0;
      double specificity = 0;
      double falsePositiveRate = 0;
      double falseNegativeRate = 0;
      double jaccard = 0;
      double dice = 0;
      double conformity = 0.0;
      
      unsigned long int truePositive = 0;
      unsigned long int trueNegative = 0;
      unsigned long int falsePositive = 0;
      unsigned long int falseNegative = 0;

      groundTruthIterator.GoToBegin();
      segmentedImageIterator.GoToBegin();
      if (outputFpFilename.length() > 0)
        falsePositiveImageIterator->GoToBegin(); 
      if (outputFnFilename.length() > 0)
        falseNegativeImageIterator->GoToBegin(); 

      while(!groundTruthIterator.IsAtEnd() && !segmentedImageIterator.IsAtEnd())
        {
          groundTruthValue = groundTruthIterator.Get();
          segmentedImageValue = segmentedImageIterator.Get();
          bool isIgnore = false; 
          
          if (ignoreSlice >= 0)
          {
            if (ignoreDirection == 'g')
            {
              if (segmentedImageIterator.GetIndex()[ignoreAxis] > ignoreSlice)
              {
                isIgnore = true; 
              }
            }
            else
            {
              if (segmentedImageIterator.GetIndex()[ignoreAxis] < ignoreSlice)
              {
                isIgnore = true; 
              }
            }
          }
          
          if (!isIgnore)
          {
            if ( groundTruthValue &&  segmentedImageValue) truePositive++;
            if (!groundTruthValue && !segmentedImageValue) trueNegative++;
            if (!groundTruthValue &&  segmentedImageValue) 
            {
              falsePositive++;
              if (outputFpFilename.length() > 0)
                falsePositiveImageIterator->Set(1); 
            }
            if ( groundTruthValue && !segmentedImageValue) 
            {
              falseNegative++;
              if (outputFnFilename.length() > 0)
                falseNegativeImageIterator->Set(1); 
            }
          }
          
          ++groundTruthIterator;
          ++segmentedImageIterator;
          if (outputFpFilename.length() > 0)
            ++(*falsePositiveImageIterator); 
          if (outputFnFilename.length() > 0)
            ++(*falseNegativeImageIterator); 
        }

      sensitivity       = ((double)truePositive)/((double)truePositive + (double)falseNegative);
      specificity       = ((double)trueNegative)/((double)trueNegative + (double)falsePositive);
      falsePositiveRate = 1.0 - specificity;
      falseNegativeRate = 1.0 - sensitivity;
      jaccard           = ((double)truePositive)/((double)truePositive + (double)falsePositive + (double)falseNegative);
      dice              = ((double)truePositive)/(0.5*((double)truePositive + (double)falseNegative + (double)truePositive + (double)falsePositive));
      if (truePositive > 0)
      {
        conformity = 1.0 - (((double)falsePositive)+((double)falseNegative))/((double)truePositive); 
      }
      
      if (outputImageName) 
        {
          std::cout << exampleSegmentedImageFileName << "\t";  
        }
      std::cout << niftk::ConvertToString((int)truePositive) \
        << "\t" << niftk::ConvertToString((int)trueNegative) \
        << "\t" << niftk::ConvertToString((int)falsePositive) \
        << "\t" << niftk::ConvertToString((int)falseNegative) \
        << "\t" << niftk::ConvertToString((double)sensitivity) \
        << "\t" << niftk::ConvertToString((double)specificity) \
        << "\t" << niftk::ConvertToString((double)falsePositiveRate) \
        << "\t" << niftk::ConvertToString((double)falseNegativeRate) \
        << "\t" << niftk::ConvertToString((double)jaccard) \
        << "\t" << niftk::ConvertToString((double)dice) \
        << "\t" << niftk::ConvertToString((double)conformity) \
        << std::endl;
    }
    
  // Tidy up and save the images. 
  if (outputFpFilename.length() > 0)
    delete falsePositiveImageIterator;  
  if (outputFnFilename.length() > 0)
    delete falseNegativeImageIterator;  
  ImageWriterType::Pointer writer = ImageWriterType::New(); 
  if (outputFpFilename.length() > 0)
  {
    writer->SetInput(falsePositiveImageReader->GetOutput()); 
    writer->SetFileName(outputFpFilename); 
    writer->Update(); 
  }
  if (outputFnFilename.length() > 0)
  {
    writer->SetInput(falseNegativeImageReader->GetOutput()); 
    writer->SetFileName(outputFnFilename); 
    writer->Update(); 
  }
  
  return EXIT_SUCCESS;
}

