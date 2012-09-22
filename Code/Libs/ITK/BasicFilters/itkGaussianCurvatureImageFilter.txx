/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-05-28 22:05:02 +0100 (Fri, 28 May 2010) $
 Revision          : $Revision: 3326 $
 Last modified by  : $Author: mjc $
 
 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details. 

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef __itkGaussianCurvatureImageFilter_txx
#define __itkGaussianCurvatureImageFilter_txx

#include "itkGaussianCurvatureImageFilter.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkImageRegionIterator.h"

namespace itk
{

template <typename TInputImage, typename TOutputImage>
GaussianCurvatureImageFilter<TInputImage, TOutputImage>
::GaussianCurvatureImageFilter()
{
}

template <typename TInputImage, typename TOutputImage>
GaussianCurvatureImageFilter<TInputImage, TOutputImage>
::~GaussianCurvatureImageFilter()
{
}

template <typename TInputImage, typename TOutputImage>
void
GaussianCurvatureImageFilter<TInputImage, TOutputImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
}

template <typename TInputImage, typename TOutputImage>
void
GaussianCurvatureImageFilter<TInputImage, TOutputImage>
::ThreadedGenerateData(const ImageRegionType& outputRegionForThread, int threadNumber) 
{

  TInputImage  *inputImage  = static_cast< TInputImage  * >(this->ProcessObject::GetInput(0));
  TOutputImage *outputImage = static_cast< TOutputImage * >(this->ProcessObject::GetOutput(0));

  ImageRegionType actualRegion = this->CheckAndAdjustRegion(outputRegionForThread, inputImage);
    
  ImageRegionConstIteratorWithIndex<TInputImage> inputIterator = ImageRegionConstIteratorWithIndex<TInputImage>(inputImage, actualRegion);
  ImageRegionIterator<TOutputImage> outputIterator = ImageRegionIterator<TOutputImage>(outputImage, actualRegion);
  
  PixelType gaussianCurvature;
  IndexType voxelIndex;
  
  for (inputIterator.GoToBegin(), 
       outputIterator.GoToBegin();
       !inputIterator.IsAtEnd() && !outputIterator.IsAtEnd();
       ++inputIterator,
       ++outputIterator)
  {
    voxelIndex = inputIterator.GetIndex();
    
    // At the moment im simply writing the first derivative in y,
    // so this should be swapped for a real Gaussian curvature calculation.
    
    PixelType dx = this->d(0, voxelIndex, inputImage);
    PixelType dy = this->d(1, voxelIndex, inputImage);
    PixelType dz = this->d(2, voxelIndex, inputImage);
    PixelType ddxx = this->dd(0, voxelIndex, inputImage);  
    PixelType ddyy = this->dd(1, voxelIndex, inputImage);
    PixelType ddzz = this->dd(2, voxelIndex, inputImage);
    PixelType ddxy = this->dd(0, 1, voxelIndex, inputImage);
    PixelType ddxz = this->dd(0, 2, voxelIndex, inputImage);
    PixelType ddyz = this->dd(1, 2, voxelIndex, inputImage);
    
    gaussianCurvature = pow((double)dx,2.0) * ( ddyy * ddzz - pow((double)ddyz,2) );
    gaussianCurvature += pow((double)dy,2.0) * ( ddxx * ddzz - pow((double)ddxz,2) );
    gaussianCurvature += pow((double)dz,2.0) * ( ddxx * ddyy - pow((double)ddxy,2) );
    gaussianCurvature += 2.0 * dx * dy * (ddxz * ddyz - ddxy * ddzz);
    gaussianCurvature += 2.0 * dy * dz * (ddxy * ddxz - ddyz * ddxx);
    gaussianCurvature += 2.0 * dx * dz * (ddxy * ddyz - ddxz * ddyy);
    gaussianCurvature = gaussianCurvature /  pow( (pow((double)dz,2.0) + pow((double)dy,2.0) + pow((double)dx,2.0)) , 2.0 );
        
    outputIterator.Set(gaussianCurvature);
  }
}

} // end namespace

#endif
