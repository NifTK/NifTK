/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkMeanCurvatureImageFilter_txx
#define __itkMeanCurvatureImageFilter_txx

#include "itkMeanCurvatureImageFilter.h"
#include <itkImageRegionConstIteratorWithIndex.h>
#include <itkImageRegionIterator.h>
#include <math.h>

namespace itk
{

template <typename TInputImage, typename TOutputImage>
MeanCurvatureImageFilter<TInputImage, TOutputImage>
::MeanCurvatureImageFilter()
{
}

template <typename TInputImage, typename TOutputImage>
MeanCurvatureImageFilter<TInputImage, TOutputImage>
::~MeanCurvatureImageFilter()
{
}

template <typename TInputImage, typename TOutputImage>
void
MeanCurvatureImageFilter<TInputImage, TOutputImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
}

template <typename TInputImage, typename TOutputImage>
void
MeanCurvatureImageFilter<TInputImage, TOutputImage>
::ThreadedGenerateData(const ImageRegionType& outputRegionForThread, int threadNumber) 
{

  TInputImage  *inputImage  = static_cast< TInputImage  * >(this->ProcessObject::GetInput(0));
  TOutputImage *outputImage = static_cast< TOutputImage * >(this->ProcessObject::GetOutput(0));

  ImageRegionType actualRegion = this->CheckAndAdjustRegion(outputRegionForThread, inputImage);
    
  ImageRegionConstIteratorWithIndex<TInputImage> inputIterator = ImageRegionConstIteratorWithIndex<TInputImage>(inputImage, actualRegion);
  ImageRegionIterator<TOutputImage> outputIterator = ImageRegionIterator<TOutputImage>(outputImage, actualRegion);
  
  PixelType meanCurvature;
  IndexType voxelIndex;
  
  for (inputIterator.GoToBegin(), 
       outputIterator.GoToBegin();
       !inputIterator.IsAtEnd() && !outputIterator.IsAtEnd();
       ++inputIterator,
       ++outputIterator)
  {
    voxelIndex = inputIterator.GetIndex();
    
    // Equation reference: Level Sets and Fast Marching Methods, J.A.Sethian, p70
    
    PixelType dx = this->d(0, voxelIndex, inputImage);
    PixelType dy = this->d(1, voxelIndex, inputImage);
    PixelType dz = this->d(2, voxelIndex, inputImage);
    PixelType ddxx = this->dd(0, voxelIndex, inputImage);  
    PixelType ddyy = this->dd(1, voxelIndex, inputImage);
    PixelType ddzz = this->dd(2, voxelIndex, inputImage);
    PixelType ddxy = this->dd(0, 1, voxelIndex, inputImage);
    PixelType ddxz = this->dd(0, 2, voxelIndex, inputImage);
    PixelType ddyz = this->dd(1, 2, voxelIndex, inputImage);
    
    meanCurvature = (ddyy + ddzz) * pow(dx,2) + (ddxx + ddzz) * pow(dy,2) + (ddxx + ddyy) * pow(dz,2);
    meanCurvature += - (2 * dx * dy * ddxy) - (2 * dx * dz * ddxz) - (2 * dy * dz * ddyz);
    meanCurvature = meanCurvature / pow( (pow((double)dx,2.0) + pow((double)dy,2.0) + pow((double)dz,2.0)) , 3.0/2.0);
    meanCurvature /= -2.0;
        
    
        
    outputIterator.Set(meanCurvature);
  }
}

} // end namespace

#endif
