/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "ConversionUtils.h"
#include "CommandLineParser.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionIterator.h"
#include "itkLogImageFilter.h"
#include "itkSubtractImageFilter.h"
#include "itkMedianImageFilter.h"
#include "itkExpImageFilter.h"
#include "itkDivideByConstantImageFilter.h"
#include "itkMultiplyImageFilter.h"
#include "itkDivideImageFilter.h"

/*!
 * \file niftkDbc.cxx
 * \page niftkDbc
 * \section niftkDbcSummary Implements "Correction of differential intensity inhomogeneity in longitudinal MR images", Lewis and Fox, 2004, NeuroImage. 
 *
 * This program implements "Correction of differential intensity inhomogeneity in longitudinal MR images", Lewis and Fox, 2004, NeuroImage. 
 * \li Dimensions: 3
 * \li Pixel type: Scalars only, of unsigned char, char, unsigned short, short, unsigned int, int, unsigned long, long, float. 
 *
 * \section niftkDbcCaveat Caveats
 * \li File sizes not checked.
 * \li Image headers not checked. By "voxel by voxel basis" we mean that the image geometry, origin, orientation is not checked.
 */

struct niftk::CommandLineArgumentDescription clArgList[] = {

  {OPT_STRING|OPT_REQ, "i1",  "filename", "Input image 1."},

  {OPT_STRING, "m1",  "filename", "Input mask for image 1."},
  
  {OPT_STRING|OPT_REQ, "i2",  "filename", "Input image 2."},
  
  {OPT_STRING, "m2",  "filename", "Input mask for image 2."},
  
  {OPT_STRING|OPT_REQ, "o1", "filename", "Output image 1."},
  
  {OPT_STRING|OPT_REQ, "o2", "filename", "Output image 2."},
  
  {OPT_INT, "radius", "value", "Radius of the median filter."},

  {OPT_DONE, NULL, NULL, "Perform the differential bias correction on the two images."}
   
};


enum {
  O_INPUT_FILE1=0,
  
  O_INPUT_MASK1,
  
  O_INPUT_FILE2,
  
  O_INPUT_MASK2,
  
  O_OUTPUT_FILE1, 
  
  O_OUTPUT_FILE2, 
  
  O_INT_RADIUS

};

/**
 * \brief Differential bias correction. 
 * 
 * Implements "Correction of differential intensity inhomogeneity in longitudinal MR images", Lewis and Fox, 2004, NeuroImage. 
 * 
 */
int main(int argc, char** argv)
{
  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);
  std::string inputFilename1;
  std::string inputFilename2;
  std::string inputMaskFilename1;
  std::string inputMaskFilename2; 
  std::string outputFilename1;
  std::string outputFilename2;
  int inputRadius = 5; 
  
  CommandLineOptions.GetArgument(O_INPUT_FILE1, inputFilename1);
  CommandLineOptions.GetArgument(O_INPUT_MASK1, inputMaskFilename1);
  CommandLineOptions.GetArgument(O_INPUT_FILE2, inputFilename2);
  CommandLineOptions.GetArgument(O_INPUT_MASK2, inputMaskFilename2);
  CommandLineOptions.GetArgument(O_OUTPUT_FILE1, outputFilename1);
  CommandLineOptions.GetArgument(O_OUTPUT_FILE2, outputFilename2);
  CommandLineOptions.GetArgument(O_INT_RADIUS, inputRadius); 
  
  typedef float PixelType; 
  typedef short MaskPixelType; 
  const int Dimension = 3; 
  typedef itk::Image<PixelType, Dimension>  InputImageType; 
  typedef itk::Image<MaskPixelType, Dimension>  InputMaskType; 
  typedef itk::ImageFileReader<InputImageType> ReaderType;
  typedef itk::ImageFileReader<InputMaskType> MaskReaderType;
  typedef itk::ImageFileWriter<InputImageType> WriterType;
  typedef itk::ImageRegionIterator<InputImageType> ImageIterator; 
  typedef itk::ImageRegionIterator<InputMaskType> MaskIterator; 
  typedef itk::LogImageFilter<InputImageType, InputImageType> LogImageFilterType; 
  typedef itk::SubtractImageFilter<InputImageType, InputImageType> SubtractImageFilterType; 
  typedef itk::MedianImageFilter<InputImageType, InputImageType> MedianImageFilterType; 
  typedef itk::ExpImageFilter<InputImageType, InputImageType> ExpImageFilterType; 
  typedef itk::DivideByConstantImageFilter<InputImageType, PixelType, InputImageType> DivideByConstantImageFilterType; 
  typedef itk::MultiplyImageFilter<InputImageType, InputImageType> MultiplyImageFilterType; 
  typedef itk::DivideImageFilter<InputImageType, InputImageType, InputImageType> DivideImageFilterType; 
  
  try
  {
    ReaderType::Pointer inputReader1 = ReaderType::New();
    ReaderType::Pointer inputReader2 = ReaderType::New();
    MaskReaderType::Pointer inputMaskReader1 = MaskReaderType::New();
    MaskReaderType::Pointer inputMaskReader2 = MaskReaderType::New();
    WriterType::Pointer writer = WriterType::New();

    inputReader1->SetFileName(inputFilename1);
    inputReader1->Update(); 
    inputReader2->SetFileName(inputFilename2);
    inputReader2->Update(); 
    inputMaskReader1->SetFileName(inputMaskFilename1);
    inputMaskReader1->Update(); 
    inputMaskReader2->SetFileName(inputMaskFilename2);
    inputMaskReader2->Update(); 
    
    // Normalisation. Scale the images to the geometric mean of the mean brain intensities of the two images. 
    double mean1 = 0.0; 
    double count1 = 0.0; 
    ImageIterator image1It(inputReader1->GetOutput(), inputReader1->GetOutput()->GetLargestPossibleRegion());
    MaskIterator mask1It(inputMaskReader1->GetOutput(), inputMaskReader1->GetOutput()->GetLargestPossibleRegion());; 
    for (image1It.GoToBegin(), mask1It.GoToBegin(); 
         !image1It.IsAtEnd(); 
         ++image1It, ++mask1It)
    {
      if (mask1It.Get() > 0)
      {
        mean1 += image1It.Get(); 
        count1++; 
      }
    }
    mean1 /= count1; 
    double mean2 = 0.0; 
    double count2 = 0.0; 
    ImageIterator image2It(inputReader2->GetOutput(), inputReader2->GetOutput()->GetLargestPossibleRegion());
    MaskIterator mask2It(inputMaskReader2->GetOutput(), inputMaskReader2->GetOutput()->GetLargestPossibleRegion());; 
    for (image2It.GoToBegin(), mask2It.GoToBegin(); 
         !image2It.IsAtEnd(); 
         ++image2It, ++mask2It)
    {
      if (mask2It.Get() > 0)
      {
        mean2 += image2It.Get(); 
        count2++; 
      }
    }
    mean2 /= count2; 
    double normalisedMean = sqrt(mean1*mean2); 
    for (image1It.GoToBegin(); 
        !image1It.IsAtEnd(); 
        ++image1It)
    {
      double normalisedValue = normalisedMean*image1It.Get()/mean1; 
      if (normalisedValue < 1.0)
        normalisedValue = 1.0; 
      image1It.Set(normalisedValue); 
    }
    for (image2It.GoToBegin(); 
        !image2It.IsAtEnd(); 
        ++image2It)
    {
      double normalisedValue = normalisedMean*image2It.Get()/mean2; 
      if (normalisedValue < 1.0)
        normalisedValue = 1.0; 
      image2It.Set(normalisedValue); 
    }
    
    // Take log. 
    LogImageFilterType::Pointer logImageFilter1 = LogImageFilterType::New(); 
    LogImageFilterType::Pointer logImageFilter2 = LogImageFilterType::New(); 
    
    logImageFilter1->SetInput(inputReader1->GetOutput()); 
    logImageFilter2->SetInput(inputReader2->GetOutput()); 
    
    // Subtract the log images.     
    SubtractImageFilterType::Pointer subtractImageFilter = SubtractImageFilterType::New(); 
    subtractImageFilter->SetInput1(logImageFilter1->GetOutput()); 
    subtractImageFilter->SetInput2(logImageFilter2->GetOutput()); 
    
    // Apply median filter to the subtraction image.  
    MedianImageFilterType::Pointer medianImageFilter = MedianImageFilterType::New(); 
    InputImageType::SizeType kernelRadius; 
    kernelRadius.Fill(inputRadius); 
    medianImageFilter->SetInput(subtractImageFilter->GetOutput()); 
    medianImageFilter->SetRadius(kernelRadius); 
    
    // Divide the bias into two equal parts. 
    DivideByConstantImageFilterType::Pointer divideByConstantImageFilter = DivideByConstantImageFilterType::New(); 
    divideByConstantImageFilter->SetInput(medianImageFilter->GetOutput()); 
    divideByConstantImageFilter->SetConstant(2.0); 
    
    // Exponential the output from the median filter to get the bias ratio. 
    ExpImageFilterType::Pointer expImageFilter = ExpImageFilterType::New(); 
    expImageFilter->SetInput(divideByConstantImageFilter->GetOutput()); 
    
    // Apply the bias to the images. 
    // 1. multiple the image2 by the bias. 
    // 2. divide the iamge1 by the bias. 
    MultiplyImageFilterType::Pointer multiplyImageFilter = MultiplyImageFilterType::New(); 
    multiplyImageFilter->SetInput1(inputReader2->GetOutput()); 
    multiplyImageFilter->SetInput2(expImageFilter->GetOutput()); 
    
    DivideImageFilterType::Pointer divideImageFilter = DivideImageFilterType::New(); 
    divideImageFilter->SetInput1(inputReader1->GetOutput()); 
    divideImageFilter->SetInput2(expImageFilter->GetOutput()); 
    
    // Save them. 
    writer->SetFileName(outputFilename1);
    writer->SetInput(divideImageFilter->GetOutput());  
    writer->Update(); 
  
    writer->SetFileName(outputFilename2);
    writer->SetInput(multiplyImageFilter->GetOutput());  
    writer->Update(); 
  }  
  catch (itk::ExceptionObject& exceptionObject)
  {
    std::cerr << "Error:" << exceptionObject << std::endl;
    return EXIT_FAILURE; 
  }
  
  return 0; 
  
}
  
  
  
  
  
  
  
