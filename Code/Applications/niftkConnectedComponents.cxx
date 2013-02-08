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

#include "itkLogHelper.h"
#include "ConversionUtils.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkConnectedComponentImageFilter.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"

/*!
 * \file niftkConnectedComponents.cxx
 * \page niftkConnectedComponents
 * \section niftkConnectedComponentsSummary Runs ITK ConnectedComponentImageFilter to find connected components.
 *
 * This program runs ITK ConnectedComponentImageFilter to find connected components.
 * \li Dimensions: 3
 * \li Pixel type: Scalars only, of unsigned char, char, short. 
 *
 */
void StartUsage(char *name)
{
  niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
  std::cout << "  " << std::endl;
  std::cout << "  Runs ITK ConnectedComponentImageFilter to find connected components. " << std::endl;
  std::cout << "  " << std::endl;
  std::cout << "  " << name << " <input> <output_prefix> <output_ext> -largest -background background -verbose" << std::endl;
  std::cout << "  " << std::endl;
  std::cout << "*** [mandatory] ***" << std::endl << std::endl;
  std::cout << "  <input>            Input file" << std::endl;
  std::cout << "  <output_prefix>    Output file prefix" << std::endl;
  std::cout << "  <output_ext>       Output file extension" << std::endl;
  std::cout << std::endl;
  std::cout << "*** [optional] ***" << std::endl << std::endl;
  std::cout << "  <-largest>         Specifiy to only save the largest component. All components are saved by default." << std::endl;
  std::cout << "  <-background>      Specifiy the background value of the output image. [0]" << std::endl;
  std::cout << "  <-verbose>         More output. No by default" << std::endl;
  std::cout << std::endl;
}



int main(int argc, char** argv)
{
  const unsigned int Dimension = 3;
  typedef short InputPixelType;
  typedef short OutputPixelType;
  
  std::string inputImageName; 
  std::string outputImagePrefixName; 
  std::string outputImageExtName; 
  bool isOnlySaveLargest = false; 
  InputPixelType backgroundValue = 0; 
  bool isVerbose = false; 
  bool isFullyConnected = false;

  if (argc < 3)
  {
    StartUsage(argv[0]); 
    return EXIT_FAILURE; 
  }

  

  int argIndex = 1; 
  inputImageName = argv[argIndex];
  std::cout << "input=" << inputImageName<< std::endl;
  argIndex += 1; 
  outputImagePrefixName = argv[argIndex]; 
  std::cout << "output_prefix=" << outputImagePrefixName<< std::endl;
  argIndex += 1; 
  outputImageExtName = argv[argIndex]; 
  std::cout << "output_ext=" << outputImageExtName<< std::endl;

  for (int i=4; i < argc; i++)
  {
    if (strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 || strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--h")==0)
    {
      StartUsage(argv[0]);
      return -1;
    }
    else if(strcmp(argv[i], "-fullyconnected") == 0)
    {
      isFullyConnected = true; 
      std::cout << "Set -largest"<< std::endl;
    }
    else if(strcmp(argv[i], "-largest") == 0)
    {
      isOnlySaveLargest = true; 
      std::cout << "Set -largest"<< std::endl;
    }
    else if(strcmp(argv[i], "-background") == 0)
    {
      backgroundValue = atoi(argv[++i]);
      std::cout << "Set -backgroundValue=" << niftk::ConvertToString(backgroundValue)<< std::endl;
    }
    else if(strcmp(argv[i], "-verbose") == 0)
    {
      isVerbose = true; 
      std::cout << "Set -verbose"<< std::endl;
    }
    else 
    {
      std::cout << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      StartUsage(argv[0]);
      return -1;
    }            
  }
  

  typedef itk::Image<InputPixelType, Dimension> InputImageType;
  typedef itk::Image<OutputPixelType, Dimension> OutputImageType;
  typedef itk::ImageFileReader<InputImageType> ImageFileReaderType;
  typedef itk::ImageFileWriter<InputImageType> ImageFileWriterType;
  typedef itk::ConnectedComponentImageFilter<InputImageType, OutputImageType> ConnectedComponentImageFilterType;
  
  try
  {
    ImageFileReaderType::Pointer reader = ImageFileReaderType::New(); 
    reader->SetFileName(inputImageName); 
    reader->Update(); 
  
    ConnectedComponentImageFilterType::Pointer ccFilter = ConnectedComponentImageFilterType::New();
    ccFilter->SetInput(reader->GetOutput());
    ccFilter->SetFullyConnected(isFullyConnected);
    ccFilter->SetBackgroundValue(backgroundValue); 
    ccFilter->Update(); 
    
    ConnectedComponentImageFilterType::LabelType numberOfComponents = ccFilter->GetObjectCount(); 
    std::cout << "Number of connected components found=" << numberOfComponents << std::endl; 
    
    // Count the number of voxels in each components.
    std::map<OutputPixelType, double> componentSizes; 
    double largestSize = 0; 
    OutputPixelType largestSizeLabel = 0; 
    typedef itk::ImageRegionConstIterator<OutputImageType> ImageRegionConstIteratorType;
    ImageRegionConstIteratorType ccIt(ccFilter->GetOutput(), ccFilter->GetOutput()->GetLargestPossibleRegion());
    for (ccIt.GoToBegin(); !ccIt.IsAtEnd(); ++ccIt)
    {
      if (ccIt.Get() != backgroundValue)
        componentSizes[ccIt.Get()]++; 
    }
    for (std::map<OutputPixelType, double>::iterator it = componentSizes.begin(); it != componentSizes.end(); it++)
    {
      if (isVerbose)
        std::cout << "Component " << it->first << " size=" << it->second << std::endl; 
      if (it->second > largestSize)
      {
        largestSize = it->second; 
        largestSizeLabel = it->first; 
      }
    }
    std::cout << "Largest label=" << largestSizeLabel << " with size=" << largestSize << std::endl; 
    
    std::map<OutputPixelType, double>::iterator startIterator = componentSizes.begin(); 
    std::map<OutputPixelType, double>::iterator endIterator = componentSizes.end(); 
    // If only save the largest one, go find it. 
    if (isOnlySaveLargest)
    {
      startIterator = componentSizes.find(largestSizeLabel); 
      if (startIterator != endIterator)
      {
        endIterator = startIterator; 
        endIterator++; 
      }
    }
      
    // Save each component. Write over the reader. 
    ImageFileWriterType::Pointer writer = ImageFileWriterType::New();  
    typedef itk::ImageRegionIterator<InputImageType> ImageRegionIterator;
    ImageRegionIterator outputIt(reader->GetOutput(), reader->GetOutput()->GetLargestPossibleRegion());
    for (std::map<OutputPixelType, double>::iterator it = startIterator; it != endIterator; it++)
    {
      for (ccIt.GoToBegin(), outputIt.GoToBegin(); 
           !ccIt.IsAtEnd(); 
           ++ccIt, ++outputIt)
      {
        if (ccIt.Get() == it->first)
          outputIt.Set(it->first); 
        else
          outputIt.Set(backgroundValue); 
      }
      writer->SetInput(reader->GetOutput()); 
      std::string outputImageName = outputImagePrefixName+niftk::ConvertToString(it->first)+"."+outputImageExtName; 
      if (isOnlySaveLargest)
        outputImageName = outputImagePrefixName+"."+outputImageExtName; 
      writer->SetFileName(outputImageName); 
      writer->Update(); 
    }
    
    
  } 
  catch (itk::ExceptionObject &e)
  {
    std::cout << "Exception caught:" << e << std::endl;
    return EXIT_FAILURE;
  }
  
  return EXIT_SUCCESS; 
}


