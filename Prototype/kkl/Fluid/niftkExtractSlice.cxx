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

 Original author   : m.adnan@cs.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
 
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkExtractImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"

int main(int argc, char** argv)
{
  char* inputImageName = argv[1]; 
  char* outputImageName = argv[2]; 
  int slice = atoi(argv[3]); 
  
  typedef itk::Image<int, 3> InputImageType3D;
  typedef itk::Image<int, 2> InputImageType2D;
  typedef itk::Image<unsigned char, 2> OutputImageType;
  typedef itk::ImageFileReader<InputImageType3D> InputImageReaderType;
  typedef itk::ImageFileWriter<OutputImageType> OutputImageWriterType;
  typedef itk::ExtractImageFilter<InputImageType3D, InputImageType2D> ExtractImageFilterType; 
  typedef itk::RescaleIntensityImageFilter<InputImageType2D, OutputImageType> RescalerType;
  
  std::cerr << "Reading " << inputImageName << "..."; 
  InputImageReaderType::Pointer imageReader = InputImageReaderType::New();
  imageReader->SetFileName(inputImageName);
  imageReader->Update(); 
  std::cerr << "done" << std::endl; 
  
  ExtractImageFilterType::Pointer extractImageFilter = ExtractImageFilterType::New(); 
  extractImageFilter->SetInput(imageReader->GetOutput()); 
  InputImageType3D::RegionType region = imageReader->GetOutput()->GetLargestPossibleRegion();  
  region.SetIndex(0, slice); 
  region.SetSize(0, 0); 
  extractImageFilter->SetExtractionRegion(region); 
  
  RescalerType::Pointer rescaleFilter = RescalerType::New(); 
  rescaleFilter->SetInput(extractImageFilter->GetOutput()); 
  rescaleFilter->SetOutputMinimum(0);
  rescaleFilter->SetOutputMaximum(255);

  std::cerr << "Writing " << outputImageName << "..."; 
  OutputImageWriterType::Pointer writer = OutputImageWriterType::New(); 
  writer->SetInput(rescaleFilter->GetOutput()); 
  writer->SetFileName(outputImageName); 
  writer->Update(); 
  std::cerr << "done" << std::endl; 
  
  return 0; 
}


