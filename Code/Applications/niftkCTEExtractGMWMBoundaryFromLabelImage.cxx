/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-08-11 08:28:23 +0100 (Wed, 11 Aug 2010) $
 Revision          : $Revision: 3647 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#include "itkLogHelper.h"
#include "ConversionUtils.h"
#include "itkCommandLineHelper.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkConnectedComponentImageFilter.h"
#include "itkMinimumMaximumImageCalculator.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkImageRegionIterator.h"
#include "itkTwinThresholdBoundaryFilter.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include <algorithm>

/*!
 * \file niftkCTEExtractGMWMBoundaryFromLabelImage.cxx
 * \page niftkCTEExtractGMWMBoundaryFromLabelImage
 * \section niftkCTEExtractGMWMBoundaryFromLabelImageSummary From a given label image, assumed to have 3 values, one for GM, one for WM and one for CSF, will extract the GM/WM boundary.
 */

void Usage(char *name)
{
  niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
  std::cout << "  " << std::endl;
  std::cout << "  From a given label image, assumed to have 3 values, one for GM, one for WM and one for CSF, will extract the GM/WM boundary." << std::endl;
  std::cout << "  " << std::endl;
  std::cout << "  " << name << " -i <filename> -o <filename> [options] " << std::endl;
  std::cout << "  " << std::endl;
  std::cout << "*** [mandatory] ***" << std::endl << std::endl;
  std::cout << "    -i    <filename>        Input label image containing exactly 3 values for GM, WM, CSF." << std::endl;
  std::cout << "    -o    <filename>        Output mask image" << std::endl << std::endl;
  std::cout << "*** [options]   ***" << std::endl << std::endl;
  std::cout << "    -w    <float> [1]       Label for white matter" << std::endl;
  std::cout << "    -g    <float> [2]       Label for grey matter" << std::endl;
  std::cout << "    -fc                     Do fully connected, i.e. 8 neighbourhood in 2D and 26 neighbourhood in 3D" << std::endl;
}

struct arguments
{
  std::string inputImage;
  std::string outputImage;
  int grey;
  int white;
  int csf;
  bool fullyConnected;
};

template <int Dimension>
int DoMain(arguments args)
{
  typedef short int PixelType;

  typedef typename itk::Image< PixelType, Dimension >          InputImageType;
  typedef typename itk::Image< PixelType, Dimension >          OutputImageType;
  typedef typename itk::ImageFileReader< InputImageType  >     ImageReaderType;
  typedef typename itk::ImageFileWriter< OutputImageType >     ImageWriterType;

  typename ImageReaderType::Pointer  labelReader  = ImageReaderType::New();
  labelReader->SetFileName(  args.inputImage );
  

  try
    {
      std::cout << "Loading label image:" + args.inputImage << std::endl;
      labelReader->Update();
      std::cout << "Done" << std::endl;

    }
  catch( itk::ExceptionObject & err )
    {
      std::cerr <<"ExceptionObject caught !";
      std::cerr << err << std::endl;
      return -2;
    }

  // First we create an image with just the grey matter.
  typedef itk::BinaryThresholdImageFilter<InputImageType, InputImageType> ThresholdFilterType;
  typename ThresholdFilterType::Pointer thresholdGreyMatterFilter = ThresholdFilterType::New();
  thresholdGreyMatterFilter->SetInput(labelReader->GetOutput());
  thresholdGreyMatterFilter->SetOutsideValue(0);
  thresholdGreyMatterFilter->SetInsideValue(1);
  thresholdGreyMatterFilter->SetUpperThreshold(args.grey);
  thresholdGreyMatterFilter->SetLowerThreshold(args.grey);

  // We do connected component analysis to pull out the biggest region (i.e. remove disconnected blobs).

  std::cout << "Doing connected component analysis" << std::endl;

  typedef itk::ConnectedComponentImageFilter<InputImageType, InputImageType> ConnectedFilterType;
  typename ConnectedFilterType::Pointer connectedFilter = ConnectedFilterType::New();
  connectedFilter->SetInput(thresholdGreyMatterFilter->GetOutput());
  connectedFilter->UpdateLargestPossibleRegion();

  // The connected component filter outputs all regions, with a different label number
  // So, we need to find the label number with the most number of voxels.

  std::cout << "Calculating minium and maximum labels (which ITK makes each connected component have a contiguous label)" << std::endl;

  typedef itk::MinimumMaximumImageCalculator<InputImageType> MinMaxCalculatorType;
  typename MinMaxCalculatorType::Pointer minMaxCalculator = MinMaxCalculatorType::New();
  minMaxCalculator->SetImage(connectedFilter->GetOutput());
  minMaxCalculator->Compute();

  PixelType min = minMaxCalculator->GetMinimum();
  PixelType max = minMaxCalculator->GetMaximum();

  std::cout << "After connected components minimum label=" << niftk::ConvertToString(min) << ", maximum label=" << niftk::ConvertToString(max) << std::endl;

  unsigned long int labels = max-min+1;

  std::cout << "Number of labels=" << niftk::ConvertToString((int)labels) << std::endl;

  unsigned long int* counts = new unsigned long int[labels];
  for (unsigned long int i = 0; i < labels; i++) counts[i] = 0;

  itk::ImageRegionIterator<InputImageType> connectedIterator(connectedFilter->GetOutput(), connectedFilter->GetOutput()->GetLargestPossibleRegion());
  for (connectedIterator.GoToBegin(); !connectedIterator.IsAtEnd(); ++connectedIterator)
    {
      counts[connectedIterator.Get() - min]++;
    }

  unsigned long int largestIndex = 0;
  unsigned long int largestCounted = 0;
  for (unsigned long int i = 1; i < labels; i++) // start at 1 so we ignore background
    {
      if (counts[i] > largestCounted)
        {
          largestIndex = i;
          largestCounted = counts[i];
        }
    }
  delete [] counts;
  int largestConnectedComponentLabel = largestIndex + min;

  std::cout << "Most frequent value=" << niftk::ConvertToString((int)(largestConnectedComponentLabel)) << ", which had " << niftk::ConvertToString((int)(largestCounted)) << " voxels" << std::endl;

  // So, here we extract the largest connected component
  typename ThresholdFilterType::Pointer connectedComponentThresholdFilter = ThresholdFilterType::New();
  connectedComponentThresholdFilter->SetInput(connectedFilter->GetOutput());
  connectedComponentThresholdFilter->SetOutsideValue(0);
  connectedComponentThresholdFilter->SetInsideValue(2);
  connectedComponentThresholdFilter->SetUpperThreshold(largestConnectedComponentLabel);
  connectedComponentThresholdFilter->SetLowerThreshold(largestConnectedComponentLabel);

  // And here we extract the WM.
  typename ThresholdFilterType::Pointer thresholdWhiteMatterFilter = ThresholdFilterType::New();
  thresholdWhiteMatterFilter->SetInput(labelReader->GetOutput());
  thresholdWhiteMatterFilter->SetOutsideValue(0);
  thresholdWhiteMatterFilter->SetInsideValue(2);
  thresholdWhiteMatterFilter->SetUpperThreshold(args.white);
  thresholdWhiteMatterFilter->SetLowerThreshold(args.white);

  // This filter takes two inputs, and if input1 value is above threshold 1, and at least
  // one voxel in the surrounding neighbourhood in image 2 is above threshold 2, the output is True (1), else False (0).
  typedef itk::TwinThresholdBoundaryFilter<InputImageType> BoundaryFilterType;
  typename BoundaryFilterType::Pointer boundaryFilter = BoundaryFilterType::New();
  boundaryFilter->SetInput1(connectedComponentThresholdFilter->GetOutput());
  boundaryFilter->SetInput2(thresholdWhiteMatterFilter->GetOutput());
  boundaryFilter->SetThresholdForInput1(1);
  boundaryFilter->SetThresholdForInput2(1);
  boundaryFilter->SetTrue(1);
  boundaryFilter->SetFalse(0);
  boundaryFilter->SetFullyConnected(args.fullyConnected);
  boundaryFilter->UpdateLargestPossibleRegion();

  typename ImageWriterType::Pointer imageWriter = ImageWriterType::New();
  imageWriter->SetFileName(args.outputImage);
  imageWriter->SetInput(boundaryFilter->GetOutput());

  try
  {
    std::cout << "Saving output image:" + args.outputImage << std::endl;
    imageWriter->Update();
    std::cout << "Done" << std::endl;
  }
  catch( itk::ExceptionObject & err )
  {
    std::cerr << "Failed: " << err << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

/**
 * \brief Implements the smoothing mentioned in section 2.5 of  Acosta et. al. MIA 13 (2009) 730-743 doi:10.1016/j.media.2009.07.03
 */
int main(int argc, char** argv)
{
  // To pass around command line args
  struct arguments args;

  // Set defaults
  args.white = 1;
  args.grey = 2;
  args.fullyConnected = false;
  

  // Parse command line args
  for(int i=1; i < argc; i++){
    if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 || strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--h")==0){
      Usage(argv[0]);
      return -1;
    }
    else if(strcmp(argv[i], "-i") == 0){
      args.inputImage=argv[++i];
      std::cout << std::cout << "Set -i=" << args.inputImage << std::endl;
    }
    else if(strcmp(argv[i], "-o") == 0){
      args.outputImage=argv[++i];
      std::cout << std::cout << "Set -o=" << args.outputImage << std::endl;
    }
    else if(strcmp(argv[i], "-g") == 0){
      args.grey=atoi(argv[++i]);
      std::cout << std::cout << "Set -g=" << niftk::ConvertToString(args.grey) << std::endl;
    }
    else if(strcmp(argv[i], "-w") == 0){
      args.white=atoi(argv[++i]);
      std::cout << std::cout << "Set -w=" << niftk::ConvertToString(args.white) << std::endl;
    }
    else if(strcmp(argv[i], "-fc") == 0){
      args.fullyConnected=true;
      std::cout << std::cout << "Set -fc=" << niftk::ConvertToString(args.fullyConnected) << std::endl;
    }
    else {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return EXIT_FAILURE;
    }
  }

  // Validate command line args
  if (args.inputImage.length() == 0 || args.outputImage.length() == 0)
    {
      Usage(argv[0]);
      return EXIT_FAILURE;
    }

  int dims = itk::PeekAtImageDimension(args.inputImage);
  int result;

  switch ( dims )
    {
      case 2:
        result = DoMain<2>(args);
        break;
      case 3:
        result = DoMain<3>(args);
      break;
      default:
        std::cout << "Unsuported image dimension" << std::endl;
        exit( EXIT_FAILURE );
    }
  return result;


}
