/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-05-28 18:04:05 +0100 (Fri, 28 May 2010) $
 Revision          : $Revision: 3325 $
 Last modified by  : $Author: mjc $

 Original author   : leung@drc.ion.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkNaryAddImageFilter.h"
#include "itkCastImageFilter.h"
#include "ConversionUtils.h"

int main(int argc, char* argv[])
{
  const unsigned int Dimension = 3;
  typedef float PixelType;
  typedef short MaskPixelType; 
  typedef short OutputPixelType; 
  typedef itk::Image< PixelType, Dimension > FixedImageType;
  typedef itk::Image< MaskPixelType, Dimension > MaskImageType;
  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;
  typedef itk::ImageFileReader< FixedImageType > FixedImageReaderType;
  typedef itk::ImageFileReader< MaskImageType > MaskImageReaderType;
  typedef itk::ImageFileWriter< OutputImageType > FixedImageWriterType;
  typedef itk::NaryAddImageFilter<FixedImageType, FixedImageType> AddImageFilterType; 
  typedef itk::CastImageFilter<FixedImageType, OutputImageType> CastImageFilterType; 
  AddImageFilterType::Pointer filter = AddImageFilterType::New(); 
  
  FixedImageWriterType::Pointer writer = FixedImageWriterType::New(); 
  writer->SetFileName(argv[1]); 
  
  MaskImageReaderType::Pointer maskImageReader = MaskImageReaderType::New(); 
  maskImageReader->SetFileName(argv[2]); 
  maskImageReader->Update(); 
  itk::ImageRegionConstIterator<MaskImageType> maskIterator(maskImageReader->GetOutput(), maskImageReader->GetOutput()->GetLargestPossibleRegion()); 
  
  int startingIndex = 3; 
  int numberOfImages = argc-startingIndex; 
  float *means = new float[numberOfImages]; 
  FixedImageReaderType::Pointer *imageReader = new FixedImageReaderType::Pointer[numberOfImages]; 
  
  // Calculate the means of the images within the mask. 
  for (int i = startingIndex; i < argc; i++)
  {
    float numberOfVoxels = 0.0; 
    
    means[i-startingIndex] = 0.0; 
    imageReader[i-startingIndex] = FixedImageReaderType::New(); 
    imageReader[i-startingIndex]->SetFileName(argv[i]); 
    imageReader[i-startingIndex]->Update(); 
    
    itk::ImageRegionConstIterator<FixedImageType> iterator(imageReader[i-startingIndex]->GetOutput(), imageReader[i-startingIndex]->GetOutput()->GetLargestPossibleRegion()); 
    
    for (iterator.GoToBegin(), maskIterator.GoToBegin(); 
         !iterator.IsAtEnd(); 
         ++iterator, ++maskIterator)
    {
      if (maskIterator.Get() > 0)
      {
        means[i-startingIndex] += iterator.Get(); 
        numberOfVoxels += 1.0; 
      }
    }
    means[i-startingIndex] /= numberOfVoxels; 
    std::cout << "mean=" << means[i-startingIndex] << std::endl; 
  }
  
  float meanOfMeans = 0.0; 
  for (int i = 0; i < numberOfImages; i++)
  {
    meanOfMeans += means[i]; 
  }
  meanOfMeans /= numberOfImages; 
  std::cout << "mean of means=" << meanOfMeans << std::endl; 
  
  // Adjust the mean and sum them up. 
  for (int i = startingIndex; i < argc; i++)
  {
    itk::ImageRegionIterator<FixedImageType> iterator(imageReader[i-startingIndex]->GetOutput(), imageReader[i-startingIndex]->GetOutput()->GetLargestPossibleRegion()); 
    for (iterator.GoToBegin(), maskIterator.GoToBegin(); 
         !iterator.IsAtEnd(); 
         ++iterator, ++maskIterator)
    {
      if (maskIterator.Get() > 0)
        iterator.Set(iterator.Get()*(meanOfMeans/means[i-startingIndex])); 
    }
    
    filter->SetInput(i-startingIndex, imageReader[i-startingIndex]->GetOutput()); 
  }
  filter->Update(); 
  
  itk::ImageRegionIterator<FixedImageType> iterator(filter->GetOutput(), filter->GetOutput()->GetLargestPossibleRegion()); 
  for (iterator.GoToBegin(); !iterator.IsAtEnd(); ++iterator)
  {
    iterator.Set(niftk::Round(iterator.Get()/numberOfImages));  
  }
  
  CastImageFilterType::Pointer caster = CastImageFilterType::New(); 
  
  caster->SetInput(filter->GetOutput()); 
  writer->SetInput(caster->GetOutput()); 
  writer->Update(); 
  
  if (means != NULL) delete means;
  if (imageReader != NULL) delete imageReader;

  return 0;   
}
