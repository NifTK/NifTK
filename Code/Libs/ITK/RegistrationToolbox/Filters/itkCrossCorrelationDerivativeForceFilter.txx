/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-10-25 14:36:06 +0100 (Tue, 25 Oct 2011) $
 Revision          : $Revision: 7594 $
 Last modified by  : $Author: kkl $
 
 Original author   : leung@drc.ion.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details. 

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef __itkCrossCorrelationDerivativeForceFilter_txx
#define __itkCrossCorrelationDerivativeForceFilter_txx

#include "itkCrossCorrelationDerivativeForceFilter.h"
#include "itkConstNeighborhoodIterator.h"
#include <iomanip>

namespace itk {
  
template< class TFixedImage, class TMovingImage, class TScalar >
void
CrossCorrelationDerivativeForceFilter< TFixedImage, TMovingImage, TScalar >
::GenerateData()
{
  // Let F be the fixed image, M be the transformed moving image, 
  //   then cross correlation R = A/(BC), where
  //     A = sum((F-mF)*(M-mM)), mF and mM are the means of F and M, 
  //     B = sqrt(sum((F-mF)^2)), 
  //     C = sqrt(sum((M-mM)^2)). 
  // 
  // To calculate the derivative of R^2 as the registration force, 
  //   d(R^2)/du = 2R dR/du
  //             = 2R d(A/(BC))/du
  //             = 2R/(BC)^2 (BC dA/du - A d(BC)/du)
  //             = 2R/(BC) dA/du - 2RA/(BC)^2 d(BC)/du
  //             = 2A/(BC)^2 dA/du - 2A^2/(BC)^3 d(BC)/du   using R = A/(BC)
  //             = 2A/(BC)^2 dA/du - 2A^2/(B^2 C^3) dC/du   B (calucalted from the fixed image) doesn't depend on the deformation.
  //           
  //   dA/du = (F-mF) dM/du
  // 
  //   dC/du = 1/2C d(sum((M-mM)^2))/du
  //         = 1/C (M-mM) dM/du
  // 
  // Therefore, 
  //   d(R^2)/du = 2A/(BC)^2 (F-mF) dM/du - 4A^2/(B^2*C^4) (M-mM) dM/du
  //   d(R^2)/du = (2A/(BC)^2 (F-mF) - 4A^2/(B^2*C^4) (M-mF)) dM/du
  //
  //  Using symmetric finite difference to caluclate the dM/du, 
  //   d(R^2)/du = (2A/(BC)^2 (F-mF) - 2A^2/(B^2*C^4) (M-mF)) (M(x+1) - M(x-1))/2
  //   d(R^2)/du = (A/(BC)^2 (F-mF) - A^2/(B^2*C^4) (M-mF)) (M(x+1) - M(x-1))
  //   d(R^2)/du = (A/(BC)^2 (F-mF) - A^2/(B^2*C^4) (M-mF)) (M(x+1) - M(x-1))
  //   d(R^2)/du = (factor1*(F-mF) - factor2*(M-mF)) (M(x+1) - M(x-1)), where
  //     factor1 = (A/(BC)^2, factor2 = A^2/(B^2*C^4). 
  // 
  //
  // Compare this with the force in fluid_jo.c (Peter Freeborough's fluid registration at DRC), 
  //     factorG = 2A/(BC)^2, 
  //     factorF = 2A^2/(B^2 C^4)
  // and 
  //     constant = 0.5*(factorG*((double)(*gptr) - mG) + factorF*(p - mF));    (note: G is fixed, and F is moving here)
  //     bptr->x = -constant * p_dx;
  //     bptr->y = -constant * p_dy;
  //     bptr->z = -constant * p_dz;
  //  
  
  // Allocate space for output. 
  this->AllocateOutputs(); 
  
  // Pointer to output image.
  typename OutputImageType::Pointer forceImage = dynamic_cast<OutputImageType*>(this->ProcessObject::GetOutput(0));
  assert(forceImage.IsNotNull()); 
  OutputImageSpacingType spacing = forceImage->GetSpacing();
  
  // Pointer to fixed image.
  typename InputImageType::Pointer fixedImage = dynamic_cast<InputImageType*>(this->ProcessObject::GetInput(0));
  assert(fixedImage.IsNotNull()); 

  // Pointer to transformedMovingImage
  typename InputImageType::Pointer transformedMovingImage = dynamic_cast<InputImageType*>(this->ProcessObject::GetInput(1));
  assert(transformedMovingImage.IsNotNull()); 

  // To calculate A, B and C. 
  //   A = sum((F-mF)*(M-mM)) = sum(F*M) - N*mF*mM. 
  //   B = sqrt(sum((F-mF)^2)) = sqrt(sum(F^2) - N*mF*mF) 
  //   C = sqrt(sum((M-mM)^2)) = sqrt(sum(M^2) - N*mM*mM)
  ImageRegionConstIterator<InputImageType> simpleFixedImageIterator(fixedImage, fixedImage->GetLargestPossibleRegion());
  ImageRegionConstIterator<InputImageType> simpleMovingImageIterator(transformedMovingImage, transformedMovingImage->GetLargestPossibleRegion());
  double numberOfVoxels = 0.0; 
  // To store sum(F*M). 
  double crossTotal = 0; 
  // To store sum(M). 
  double movingTotal = 0; 
  // To store sum(F).
  double fixedTotal = 0;  
  // To store sum(M^2). 
  double movingSqaureTotal = 0; 
  // To store sum(F^2). 
  double fixedSquareTotal = 0;  
  
  for (simpleFixedImageIterator.GoToBegin(), simpleMovingImageIterator.GoToBegin(); 
       !simpleFixedImageIterator.IsAtEnd(); 
       ++simpleFixedImageIterator, ++simpleMovingImageIterator)
  {
    // Check if the voxel is inside the mask. 
    typename TFixedImage::PointType physicalPoint; 
    fixedImage->TransformIndexToPhysicalPoint(simpleFixedImageIterator.GetIndex(), physicalPoint); 
    if (this->m_FixedImageMask && !this->m_FixedImageMask->IsInside(physicalPoint))
    {
      continue; 
    }
    
    double fixedValue = static_cast<double>(simpleFixedImageIterator.Get()); 
    double movingValue = static_cast<double>(simpleMovingImageIterator.Get()); 
    
    crossTotal += fixedValue*movingValue; 
    movingTotal += movingValue; 
    fixedTotal += fixedValue; 
    movingSqaureTotal += movingValue*movingValue; 
    fixedSquareTotal += fixedValue*fixedValue; 
    numberOfVoxels++; 
  }
  
  double factorA = crossTotal - (movingTotal*fixedTotal)/numberOfVoxels; 
  double factorBSquare = fixedSquareTotal - (fixedTotal*fixedTotal)/numberOfVoxels; 
  double factorCSquare = movingSqaureTotal - (movingTotal*movingTotal)/numberOfVoxels; 
  double movingMean = movingTotal/numberOfVoxels; 
  double fixedMean = fixedTotal/numberOfVoxels; 
  niftkitkInfoMacro(<<"GenerateData(): numberOfVoxels=" << numberOfVoxels << ",factorA=" << factorA << ",factorBSquare=" << factorBSquare << ",factorCSquare=" << factorCSquare);
  
  // factor1 = A/(BC)^2. 
  double factor1 = factorA/(factorBSquare*factorCSquare); 
  // factor2 = A^2/(B^2*C^4)
  double factor2 = (factorA*factorA)/(factorBSquare*factorCSquare*factorCSquare); 
  niftkitkDebugMacro(<<"GenerateData(): R^2=" << (factorA*factorA)/(factorBSquare*factorCSquare));
  
  double factor1Backward = factor1; 
  double factor2Backward = (factorA*factorA)/(factorCSquare*factorBSquare*factorBSquare); 
  std::cout << std::setprecision(15) << "factor1=" << factor1 << ",factor2=" << factor2 << ",factor1Backward=" << factor1Backward << ",factor2Backward=" << factor2Backward << std::endl;
  
  typedef itk::ConstNeighborhoodIterator<InputImageType> NeighborhoodIteratorType;
  typename NeighborhoodIteratorType::RadiusType radius;
  radius.Fill(1);
  NeighborhoodIteratorType fixedImageIterator(radius, fixedImage, fixedImage->GetLargestPossibleRegion());
  NeighborhoodIteratorType transformedImageIterator(radius, transformedMovingImage, transformedMovingImage->GetLargestPossibleRegion());
  ImageRegionIterator<OutputImageType> forceImageIterator(forceImage, forceImage->GetLargestPossibleRegion());  
  ImageRegionIterator<typename Superclass::JacobianImageType>* movingImageTransformJacobianIterator = NULL; 
  ImageRegionIterator<typename Superclass::JacobianImageType>* fixedImageTransformJacobianIterator = NULL; 
  
  if (this->m_MovingImageTransformJacobian.IsNotNull())
  {
	niftkitkDebugMacro(<<"Initialising moving image transform iterator");
    movingImageTransformJacobianIterator = new ImageRegionIterator<typename Superclass::JacobianImageType>(this->m_MovingImageTransformJacobian, this->m_MovingImageTransformJacobian->GetLargestPossibleRegion()); 
    movingImageTransformJacobianIterator->GoToBegin(); 
  }
  if (this->m_FixedImageTransformJacobian.IsNotNull())
  {
	niftkitkDebugMacro(<<"Initialising fixed image transform iterator");
    fixedImageTransformJacobianIterator = new ImageRegionIterator<typename Superclass::JacobianImageType>(this->m_FixedImageTransformJacobian, this->m_FixedImageTransformJacobian->GetLargestPossibleRegion()); 
    fixedImageTransformJacobianIterator->GoToBegin(); 
  }
  
  // Set iterators to begining.
  fixedImageIterator.GoToBegin();
  transformedImageIterator.GoToBegin();
  forceImageIterator.GoToBegin();
  
  // To calculate force now - using symmetric finite difference on the either side of the current voxel. 
  for (; !fixedImageIterator.IsAtEnd(); ++fixedImageIterator, ++transformedImageIterator, ++forceImageIterator)
  {
    typename OutputImageType::PixelType forceImageVoxel;
    
    // Check if the voxel is inside the mask. 
    //typename TFixedImage::PointType physicalPoint; 
    //fixedImage->TransformIndexToPhysicalPoint(fixedImageIterator.GetIndex(), physicalPoint); 
    //if (this->m_FixedImageMask && !this->m_FixedImageMask->IsInside(physicalPoint))
    //{
    //  for (int dimensinIndex = 0; dimensinIndex < TFixedImage::ImageDimension; dimensinIndex++)
    //    forceImageVoxel[dimensinIndex] = 0.0;  
    //  forceImageIterator.Set(forceImageVoxel);
    //  continue; 
    //}
    double fixedImageCenter = fixedImageIterator.GetCenterPixel(); 
    double movingImageCenter = transformedImageIterator.GetCenterPixel(); 
    
    // d(R^2)/du = (factor1*(F-mF) - factor2*(M-mF)) (M(x+1) - M(x-1)). 
    double fixedImageDiff = fixedImageCenter-fixedMean; 
    double movingImageDiff = movingImageCenter-movingMean; 
    double theBigFactor = factor1*fixedImageDiff - factor2*movingImageDiff; 
    double theBigFactorBackward = factor1Backward*movingImageDiff - factor2Backward*fixedImageDiff; 
    
    for (int dimensinIndex = 0; dimensinIndex < TFixedImage::ImageDimension; dimensinIndex++)
    {
      typename NeighborhoodIteratorType::OffsetType minusOffset;
      typename NeighborhoodIteratorType::OffsetType plusOffset;
      
      minusOffset.Fill(0);
      plusOffset.Fill(0);
      minusOffset[dimensinIndex] += -1;
      plusOffset[dimensinIndex] += 1;
      double movingImageForce = -theBigFactor*
                                 (static_cast<double>(transformedImageIterator.GetPixel(plusOffset))
                                  -static_cast<double>(transformedImageIterator.GetPixel(minusOffset))); 
      
      double fixedImageForce = -theBigFactorBackward*
                                (static_cast<double>(fixedImageIterator.GetPixel(plusOffset))
                                 -static_cast<double>(fixedImageIterator.GetPixel(minusOffset)));       
      
      if (movingImageTransformJacobianIterator != NULL)
      {
        fixedImageForce *= movingImageTransformJacobianIterator->Get(); 
      }
      if (fixedImageTransformJacobianIterator != NULL)
      {
        movingImageForce *= fixedImageTransformJacobianIterator->Get(); 
      }
      
      forceImageVoxel[dimensinIndex] = movingImageForce; 
      
      if (this->m_IsSymmetric)
      {
        forceImageVoxel[dimensinIndex] = (movingImageForce-fixedImageForce)/2.; 
        
        // Debug. 
        //if ((forceImageVoxel[dimensinIndex] > 1e-15 && forceImageVoxel[dimensinIndex] < 1e-10) || (forceImageVoxel[dimensinIndex] < -1e-15 && forceImageVoxel[dimensinIndex] >- 1e-10))
        {
          //std::cout << std::setprecision(15) << transformedImageIterator.GetPixel(plusOffset) << "," << transformedImageIterator.GetPixel(minusOffset) << "," << fixedImageIterator.GetPixel(plusOffset) << "," << fixedImageIterator.GetPixel(minusOffset) << std::endl; 
          //std::cout << std::setprecision(15) << "theBigFactor=" << theBigFactor << ",theBigFactorBackward=" << theBigFactorBackward << ",movingImageForce=" << movingImageForce << ",fixedImageForce=" << fixedImageForce << "," << forceImageVoxel << std::endl; 
          //std::cout << "fixedMean=" << fixedMean << ",movingMean=" << movingMean << ",fixedImageIterator.GetCenterPixel()=" << fixedImageIterator.GetCenterPixel() << ",transformedImageIterator.GetCenterPixel()=" << transformedImageIterator.GetCenterPixel() << std::endl; 
          //std::cout << fixedImageCenter-fixedMean << "," << movingImageCenter-movingMean << "," << factor1*(fixedImageCenter-fixedMean) << "," << factor2*(movingImageCenter-movingMean) << "," << factor1Backward*(movingImageCenter-movingMean) << "," << factor2Backward*(fixedImageCenter-fixedMean) << std::endl; 
        }
      }
      
    }
    // Here we set the force image vector.
    if (this->GetScaleToSizeOfVoxelAxis())
    {
      for (int dimensinIndex = 0; dimensinIndex < TFixedImage::ImageDimension; dimensinIndex++)
      { 
        forceImageVoxel[dimensinIndex] *= spacing[dimensinIndex]; 
      }      
    }
    
    forceImageIterator.Set(forceImageVoxel);
    
    if (movingImageTransformJacobianIterator != NULL)
    {
      ++(*movingImageTransformJacobianIterator);  
    }
    if (fixedImageTransformJacobianIterator != NULL)
    {
      ++(*fixedImageTransformJacobianIterator); 
    }
    
  }
  
  delete movingImageTransformJacobianIterator;
  delete fixedImageTransformJacobianIterator; 
  
}

}

#endif

