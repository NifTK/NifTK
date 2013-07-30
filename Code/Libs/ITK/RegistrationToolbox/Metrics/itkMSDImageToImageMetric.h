/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkMSDImageToImageMetric_h
#define itkMSDImageToImageMetric_h

#include "itkJacobianGradientSimilarityMeasure.h"

namespace itk
{
/** 
 * \class MSDImageToImageMetric
 * \brief Implements Mean of Squared Difference similarity measure.
 *
 * \ingroup RegistrationMetrics
 */
template < class TFixedImage, class TMovingImage > 
class ITK_EXPORT MSDImageToImageMetric : 
    public JacobianGradientSimilarityMeasure< TFixedImage, TMovingImage>
{
public:

  /** Standard class typedefs. */
  typedef MSDImageToImageMetric                                  Self;
  typedef SimilarityMeasure<TFixedImage, TMovingImage >          Superclass;
  typedef SmartPointer<Self>                                     Pointer;
  typedef SmartPointer<const Self>                               ConstPointer;
  typedef typename Superclass::MeasureType                       MeasureType;
  typedef typename Superclass::RealType                          RealType;
  typedef typename Superclass::DerivativeType                    DerivativeType;
  typedef typename Superclass::GradientPixelType                 GradientPixelType;
  typedef typename Superclass::TransformJacobianType             TransformJacobianType;
  typedef typename Superclass::FixedImagePixelType               FixedImagePixelType;
  typedef typename Superclass::MovingImagePixelType              MovingImagePixelType;
  
  /** Method for creation through the object factory. */
  itkNewMacro(Self);
 
  /** Run-time type information (and related methods). */
  itkTypeMacro(MSDImageToImageMetric, SimilarityMeasure);

protected:
  
  MSDImageToImageMetric() {};
  virtual ~MSDImageToImageMetric() {};

  /**
   * Called at the start of each evaluation. 
   */
  void ResetCostFunction()
    { 
      this->m_MSD = 0;
      this->m_NumberOfSamplesForCostFunction = 0;
    }
  
  /** 
   * In this method, we calculate sum of squared difference.
   */
  void AggregateCostFunctionPair(
      FixedImagePixelType fixedValue, 
      MovingImagePixelType movingValue)
    {
      this->m_MSD += ((fixedValue - movingValue) * (fixedValue - movingValue));
      this->m_NumberOfSamplesForCostFunction++;
    }
  
  /**
   * In this method, we do any final aggregating, in this case none.
   */
  MeasureType FinalizeCostFunction()
    {
      if (this->m_NumberOfSamplesForCostFunction > 0)
        {
          return this->m_MSD / (double)m_NumberOfSamplesForCostFunction;    
        }
      else
        {
          return 0;
        }
    }

  /**
   * Called at the start of the derivative calculation.
   */
  void ResetDerivativeComputations()
    {
      m_NumberOfSamplesForDerivative = 0;  
    }

  /** 
   * Called repeatedly by base class to calculate derivative.
   */
  void ComputeDerivativeValue(
      DerivativeType  & derivative,
      const GradientPixelType & gradientPixel,
      const TransformJacobianType & jacobian,      
      unsigned int dimensions,
      unsigned int parameterNumber,
      RealType fixedValue, 
      RealType movingValue)
    {
      RealType sum = NumericTraits< RealType >::Zero;
      for(unsigned int dim=0; dim < dimensions; dim++)
        {
          sum += 2.0 * (movingValue - fixedValue) * jacobian( dim, parameterNumber ) * gradientPixel[dim];
        }
      derivative[parameterNumber] += sum;
      m_NumberOfSamplesForDerivative++;
    }


  /**
   * Called at the end of the derivative calcs.
   * In this case, as its MeanSquaredDifference, we divide by number of samples.
   */
  void FinalizeDerivative(DerivativeType  & derivative)
    {
      if( !this->m_NumberOfSamplesForDerivative )
        {
          itkExceptionMacro(<<"All the points mapped to outside of the moving image");
        }
      else
        {
          unsigned int parametersDimension = derivative.GetSize();
          
          for(unsigned int i = 0; i < parametersDimension; i++)
            {
              derivative[i] /= this->m_NumberOfSamplesForDerivative;
            }
        }
    }
  
private:
  MSDImageToImageMetric(const Self&); // purposefully not implemented
  void operator=(const Self&);        // purposefully not implemented
  
  /** The variables we need to sum up the values. */
  double m_MSD;
  long int m_NumberOfSamplesForCostFunction;
  long int m_NumberOfSamplesForDerivative;
};

} // end namespace itk

#endif



