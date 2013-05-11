/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef _itkJacobianGradientSimilarityMeasure_txx
#define _itkJacobianGradientSimilarityMeasure_txx

#include "itkJacobianGradientSimilarityMeasure.h"
#include <itkImageRegionConstIteratorWithIndex.h>

#include <itkLogHelper.h>

namespace itk
{
/*
 * Constructor
 */
template <class TFixedImage, class TMovingImage> 
JacobianGradientSimilarityMeasure<TFixedImage,TMovingImage>
::JacobianGradientSimilarityMeasure()
{
}

/*
 * PrintSelf
 */
template <class TFixedImage, class TMovingImage> 
void
JacobianGradientSimilarityMeasure<TFixedImage, TMovingImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf( os, indent );
}

/*
 * Get the Derivative Measure, using Jacobian of transform.
 */
template < class TFixedImage, class TMovingImage> 
void
JacobianGradientSimilarityMeasure<TFixedImage, TMovingImage>
::GetCostFunctionDerivative( const TransformParametersType & parameters, DerivativeType & derivative) const
{

  niftkitkDebugMacro(<< "Computing derivative at:" << parameters);
  
  // Reset the derivative in derived class.
  const_cast<JacobianGradientSimilarityMeasure<TFixedImage, TMovingImage>* >(this)->ResetDerivativeComputations();
  
  if( !this->GetGradientImage() )
    {
      itkExceptionMacro(<<"The gradient image is null, maybe you forgot to call Initialize()");
    }

  FixedImageConstPointer fixedImage = this->m_FixedImage;

  if( !fixedImage ) 
    {
      itkExceptionMacro( << "Fixed image has not been assigned" );
    }

  const unsigned int imageDimension = FixedImageType::ImageDimension;

  typedef  itk::ImageRegionConstIteratorWithIndex<FixedImageType> FixedIteratorType;
  typedef  itk::ImageRegionConstIteratorWithIndex<GradientImageType> GradientIteratorType;

  FixedIteratorType ti( fixedImage, this->GetFixedImageRegion() );

  typename FixedImageType::IndexType index;

  this->m_NumberOfPixelsCounted = 0;

  this->SetTransformParameters( parameters );

  const unsigned int parametersDimension = parameters.GetSize();
  derivative = DerivativeType( parametersDimension );
  derivative.Fill( NumericTraits<ITK_TYPENAME DerivativeType::ValueType>::Zero );

  ti.GoToBegin();

  while(!ti.IsAtEnd())
    {

      index = ti.GetIndex();
    
      InputPointType inputPoint;
      fixedImage->TransformIndexToPhysicalPoint( index, inputPoint );

      if( this->m_FixedImageMask && !this->m_FixedImageMask->IsInside( inputPoint ) )
        {
          ++ti;
          continue;
        }

      OutputPointType transformedPoint = this->m_Transform->TransformPoint( inputPoint );

      if( this->m_MovingImageMask && !this->m_MovingImageMask->IsInside( transformedPoint ) )
        {
          ++ti;
          continue;
        }

      if( this->m_Interpolator->IsInsideBuffer( transformedPoint ) )
        {
          const RealType fixedValue = ti.Get();
          
          if (fixedValue >= this->GetFixedLowerBound() && fixedValue <= this->GetFixedUpperBound())
            {
              const RealType movingValue = this->m_Interpolator->Evaluate( transformedPoint );
              
              if (movingValue >= this->GetMovingLowerBound() && movingValue <= this->GetMovingUpperBound())
                {
                  this->m_NumberOfPixelsCounted++;
                  const TransformJacobianType & jacobian = this->m_Transform->GetJacobian( inputPoint ); 
                  
                  // Get the gradient by NearestNeighboorInterpolation: 
                  // which is equivalent to round up the point components.
                  typedef typename OutputPointType::CoordRepType CoordRepType;
                  typedef ContinuousIndex<CoordRepType,MovingImageType::ImageDimension> MovingImageContinuousIndexType;

                  MovingImageContinuousIndexType tempIndex;
                  this->m_MovingImage->TransformPhysicalPointToContinuousIndex( transformedPoint, tempIndex );

                  typename MovingImageType::IndexType mappedIndex;
                  mappedIndex.CopyWithRound( tempIndex );

                  const GradientPixelType gradient = this->GetGradientImage()->GetPixel( mappedIndex );

                  for(unsigned int par=0; par<parametersDimension; par++)
                    {
                      // Let derived class, aggregate the value up.
                      const_cast<JacobianGradientSimilarityMeasure<TFixedImage, TMovingImage>* >(this)
                        ->ComputeDerivativeValue(
                            derivative,
                            gradient,
                            jacobian,
                            imageDimension,
                            par,
                            fixedValue,
                            movingValue);
                    }
                }
            }
        }
      ++ti;
    }

  // Let derived class, do any finalizing.
  const_cast<JacobianGradientSimilarityMeasure<TFixedImage, TMovingImage>* >(this)->FinalizeDerivative(derivative);

  niftkitkDebugMacro(<< "Computing derivative at:" << parameters << ", gives:" << derivative);
}

} // end namespace itk

#endif
