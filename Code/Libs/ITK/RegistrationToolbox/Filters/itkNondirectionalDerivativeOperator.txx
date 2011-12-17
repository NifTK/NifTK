/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-20 20:57:34 +0100 (Tue, 20 Sep 2011) $
 Revision          : $Revision: 7341 $
 Last modified by  : $Author: ad $
 
 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details. 

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef __itkNondirectionalDerivativeOperator_txx
#define __itkNondirectionalDerivativeOperator_txx
#include "itkNondirectionalDerivativeOperator.h"
#include "itkImage.h"
#include "itkImageRegionIterator.h"

#include "itkLogHelper.h"

namespace itk
{
template<class TPixel,unsigned int VDimension, class TAllocator>
NondirectionalDerivativeOperator< TPixel, VDimension, TAllocator >
::NondirectionalDerivativeOperator()
{
}

template<class TPixel,unsigned int VDimension, class TAllocator>
void
NondirectionalDerivativeOperator< TPixel, VDimension, TAllocator >
::CreateToRadius(unsigned int radius)
{
  this->SetRadius(radius);
  for (typename Superclass::Iterator iterator = this->Begin(); iterator != this->End(); ++iterator)
  {
    *iterator = 0;
  }

  typedef Image< TPixel, VDimension > FiniteDifferenceImageType; 
  typedef ImageRegionIterator< FiniteDifferenceImageType > FiniteDifferenceImageIteratorType; 
  typename FiniteDifferenceImageType::Pointer finiteDifferenceImage = FiniteDifferenceImageType::New();
  typename FiniteDifferenceImageType::SizeType imageSize; 
  typename FiniteDifferenceImageType::IndexType imageIndex; 
  
  for (unsigned int dimensionIndex = 0; dimensionIndex < VDimension; dimensionIndex++)
  {
    imageSize[dimensionIndex] = 2*radius+1; 
  }
  finiteDifferenceImage->SetRegions(imageSize);
  finiteDifferenceImage->Allocate();
  
  // Use the DerivativeOperator to generate the finite difference coefficients for each dimension
  // and apply them repeatly to the finite difference image. 
  // Sum all the derivative terms together to give the final operator. 
  
  DerivativeOperatorType directionalDerivativeOperator; 
  
  for (unsigned int index = 0; index < this->m_DervativeTermInfo.size(); index++)
  {
    const SingleDerivativeTermInfoType& termInfo = this->m_DervativeTermInfo[index]; 
    typename SingleDerivativeTermInfoType::DerivativeOrderType order = termInfo.GetDerivativeOrder();
    typename DerivativeOperatorType::SizeType singleDerivativeSize; 
    
    // 1. Initialise the image to be 0 everywhere except the centre. 
    FiniteDifferenceImageIteratorType finiteDifferenceImageIterator(finiteDifferenceImage, finiteDifferenceImage->GetLargestPossibleRegion());
    for (finiteDifferenceImageIterator.GoToBegin(); !finiteDifferenceImageIterator.IsAtEnd(); ++finiteDifferenceImageIterator)
    {
      finiteDifferenceImageIterator.Set(0);
    }
    for (unsigned int dimensionIndex = 0; dimensionIndex < VDimension; dimensionIndex++)
    {
      imageIndex[dimensionIndex] = imageSize[dimensionIndex]/2; 
    }
    finiteDifferenceImage->SetPixel(imageIndex, 1);
    
    // 2. Compute the finite difference image for each derivative term and add them to the operator. 
    singleDerivativeSize.Fill(0);
    singleDerivativeSize[0] = radius;
    for (unsigned int dimensionIndex = 0; dimensionIndex < VDimension; dimensionIndex++)
    {
      directionalDerivativeOperator.SetDirection(0);
      directionalDerivativeOperator.SetOrder(order[dimensionIndex]);
      directionalDerivativeOperator.CreateToRadius(singleDerivativeSize);
      directionalDerivativeOperator.FlipAxes();
      
      if (dimensionIndex == 0)
      {
        // x - just set the middle row to be the finite difference coefficients.
        typename FiniteDifferenceImageType::IndexType currentImageIndex; 
        for (unsigned int innerDimensionIndex = 1; innerDimensionIndex < VDimension; innerDimensionIndex++)
        {
          currentImageIndex[innerDimensionIndex] = imageSize[innerDimensionIndex]/2; 
        }
        
        typename DerivativeOperatorType::ConstIterator coefficientsIterator = directionalDerivativeOperator.Begin();
        for (unsigned int x = 0; x < imageSize[0]; x++, ++coefficientsIterator)
        {
          typename FiniteDifferenceImageType::PixelType pixel; 
          
          pixel = (*coefficientsIterator);
          currentImageIndex[0] = x;
          finiteDifferenceImage->SetPixel(currentImageIndex, pixel); 
        }
      }
      else if (dimensionIndex == 1)
      {
        // y - apply the y finite difference coefficients to the middle plane.
        typename FiniteDifferenceImageType::IndexType currentImageIndex; 
        typename FiniteDifferenceImageType::IndexType middlePixelIndex; 
        for (unsigned int innerDimensionIndex = 2; innerDimensionIndex < VDimension; innerDimensionIndex++)
        {
          currentImageIndex[innerDimensionIndex] = imageSize[innerDimensionIndex]/2; 
          middlePixelIndex[innerDimensionIndex] = imageSize[innerDimensionIndex]/2; 
        }
        
        for (unsigned int x = 0; x < imageSize[0]; x++)
        {
          typename FiniteDifferenceImageType::PixelType middlePixel; 
          
          middlePixelIndex[0] = x;
          middlePixelIndex[1] = imageSize[1]/2; 
          middlePixel = finiteDifferenceImage->GetPixel(middlePixelIndex);
          currentImageIndex[0] = x;
        
          typename DerivativeOperatorType::ConstIterator coefficientsIterator = directionalDerivativeOperator.Begin();
          for (unsigned int y = 0; y < imageSize[1]; y++, ++coefficientsIterator)
          {
            typename FiniteDifferenceImageType::PixelType pixel; 
          
            pixel = (*coefficientsIterator)*middlePixel;
            currentImageIndex[1] = y;
            finiteDifferenceImage->SetPixel(currentImageIndex, pixel);
          } 
        }
      }
      else if (dimensionIndex == 2)
      {
        // z - apply the z finite difference coefficients to the whole volume.
        typename FiniteDifferenceImageType::IndexType currentImageIndex; 
        typename FiniteDifferenceImageType::IndexType middlePixelIndex; 
        
        for (unsigned int x = 0; x < imageSize[0]; x++)
        {
          typename FiniteDifferenceImageType::PixelType middlePixel; 
          
          middlePixelIndex[0] = x;
          currentImageIndex[0] = x;
        
          for (unsigned int y = 0; y < imageSize[1]; y++)
          {
            middlePixelIndex[1] = y; 
            middlePixelIndex[2] = imageSize[2]/2; 
            middlePixel = finiteDifferenceImage->GetPixel(middlePixelIndex);
            currentImageIndex[1] = y;
          
            typename DerivativeOperatorType::ConstIterator coefficientsIterator = directionalDerivativeOperator.Begin();
            for (unsigned int z = 0; z < imageSize[2]; z++, ++coefficientsIterator)
            {
              typename FiniteDifferenceImageType::PixelType pixel; 
          
              pixel = (*coefficientsIterator)*middlePixel;
              currentImageIndex[2] = z;
              finiteDifferenceImage->SetPixel(currentImageIndex, pixel);
            }
          } 
        }
      }
      else
      {
        itkExceptionMacro("Cannot generate finite difference operator for dimension greater than 3.");
      }
    }
    
    typename Superclass::Iterator iterator;
    
    for (iterator = this->Begin(), finiteDifferenceImageIterator.GoToBegin(); 
        iterator != this->End(); 
        ++iterator, ++finiteDifferenceImageIterator)
    {
      *iterator += termInfo.GetConstant()*finiteDifferenceImageIterator.Get(); 
    }
    
  }
  
  
  
}



} // namespace


#endif









