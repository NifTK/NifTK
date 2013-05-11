/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <ConversionUtils.h>
#include <CommandLineParser.h>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkNeighborhoodConnectedImageFilter.h>
#include <itkBinaryThresholdImageFilter.h>
#include <itkVotingBinaryIterativeHoleFillingImageFilter.h>

/*!
 * \file niftkFillHoles.cxx
 * \page niftkFillHoles
 * \section niftkFillHolesSummary Runs ITK VotingBinaryIterativeHoleFillingImageFilterType to fill holes in a image.
 *
 * This program uses ITK runs ITK VotingBinaryIterativeHoleFillingImageFilterType to fill holes in a image. 
 * \li Dimensions: 3
 * \li Pixel type: Scalars only, of unsigned char, char, short
 *
 */

struct niftk::CommandLineArgumentDescription clArgList[] = {

  {OPT_STRING|OPT_REQ, "i",  "filename", "Input image."},

  {OPT_STRING|OPT_REQ, "o", "filename", "Output image."},
  
  {OPT_INT, "it", "value", "Number of iterations in the VotingBinaryIterativeHoleFillingImageFilter."},
  
  {OPT_INT, "radius", "value", "Radius of the neighborhood in the VotingBinaryIterativeHoleFillingImageFilter."},

  {OPT_DONE, NULL, NULL, "Fill hols in an image using itk::VotingBinaryIterativeHoleFillingImageFilter."}
   
};


enum {
  O_INPUT_FILE=0,

  O_OUTPUT_FILE, 
  
  O_INT_ITERATIONS, 
  
  O_INT_RADIUS

};

int main(int argc, char** argv)
{
  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);
  std::string inputFilename;
  std::string outputFilename;
  int numberOfIterations = 5;  
  int inputRadius = 2; 
  
  CommandLineOptions.GetArgument(O_INPUT_FILE, inputFilename);
  CommandLineOptions.GetArgument(O_OUTPUT_FILE, outputFilename);
  CommandLineOptions.GetArgument(O_INT_ITERATIONS, numberOfIterations);
  CommandLineOptions.GetArgument(O_INT_RADIUS, inputRadius); 
  
  typedef short PixelType; 
  const int Dimension = 3; 
  typedef itk::Image<PixelType, Dimension>  InputImageType; 
  typedef itk::ImageFileReader<InputImageType> ReaderType;
  typedef itk::ImageFileWriter<InputImageType> WriterType;
  typedef itk::NeighborhoodConnectedImageFilter<InputImageType,InputImageType> ConnectedFilterType;
  typedef itk::BinaryThresholdImageFilter<InputImageType, InputImageType> BinaryThresholdImageFilterType; 
  
  try
  {
    ReaderType::Pointer reader = ReaderType::New();
    WriterType::Pointer writer = WriterType::New();

    reader->SetFileName(inputFilename);
    writer->SetFileName(outputFilename);
    
    // Take non-zero to be the foreground and make them all to be 255. 
    BinaryThresholdImageFilterType::Pointer binaryThresholdImageFilter = BinaryThresholdImageFilterType::New(); 
    binaryThresholdImageFilter->SetInput(reader->GetOutput()); 
    binaryThresholdImageFilter->SetLowerThreshold(0); 
    binaryThresholdImageFilter->SetUpperThreshold(0); 
    binaryThresholdImageFilter->SetInsideValue(0); 
    binaryThresholdImageFilter->SetOutsideValue(255); 
    
    typedef itk::VotingBinaryIterativeHoleFillingImageFilter<InputImageType> VotingBinaryIterativeHoleFillingImageFilterType;
    VotingBinaryIterativeHoleFillingImageFilterType::Pointer filter = VotingBinaryIterativeHoleFillingImageFilterType::New();
      
    // Iteratively fill "holes" and "gaps" using VotingBinaryIterativeHoleFillingImageFilter using the 
    // user-defined number of iterations and neighborhood size. 
    // Kind of like the closing morphological operation really.
    InputImageType::SizeType indexRadius;
    indexRadius[0] = inputRadius; // radius along x
    indexRadius[1] = inputRadius; // radius along y
    indexRadius[2] = inputRadius; // radius along z
    filter->SetRadius(indexRadius);
    filter->SetBackgroundValue(0);
    filter->SetForegroundValue(255);
    filter->SetMajorityThreshold(2);
    filter->SetMaximumNumberOfIterations(numberOfIterations); 
    filter->SetInput(binaryThresholdImageFilter->GetOutput());   
    
    // Fill any holes inside objects which are disconnected from the boundary by 
    // flood filling the background from the voxel at (0,0,0). This will leave the objects and all the holes unfilled. 
    ConnectedFilterType::Pointer neighborhoodConnected = ConnectedFilterType::New();
    neighborhoodConnected->SetInput(filter->GetOutput());
    
    InputImageType::SizeType radius;
    radius[0] = 0;   
    radius[1] = 0;   
    radius[2] = 0;   
    neighborhoodConnected->SetRadius(radius);
      
    InputImageType::IndexType  index;
    index[0] = 0;
    index[1] = 0;
    index[2] = 0;
    neighborhoodConnected->SetSeed(index);
    neighborhoodConnected->SetLower(0);
    neighborhoodConnected->SetUpper(0);
    neighborhoodConnected->SetReplaceValue(255);
    
    // Make the foreground to be 255. 
    BinaryThresholdImageFilterType::Pointer binaryThresholdImageFilter1 = BinaryThresholdImageFilterType::New(); 
    binaryThresholdImageFilter1->SetInput(neighborhoodConnected->GetOutput()); 
    binaryThresholdImageFilter1->SetLowerThreshold(0); 
    binaryThresholdImageFilter1->SetUpperThreshold(0); 
    binaryThresholdImageFilter1->SetInsideValue(255); 
    binaryThresholdImageFilter1->SetOutsideValue(0); 
    
    writer->SetInput(binaryThresholdImageFilter1->GetOutput());  
    writer->Update(); 
  }  
  catch (itk::ExceptionObject& exceptionObject)
  {
    std::cerr << "Failed to load global tranform:" << exceptionObject << std::endl;
    return EXIT_FAILURE; 
  }
  
  return 0; 
  
}
  
  
  
  
  
  
  
