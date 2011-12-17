/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-14 11:37:54 +0100 (Wed, 14 Sep 2011) $
 Revision          : $Revision: 7310 $
 Last modified by  : $Author: ad $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef __itkJacobianGradientSimilarityMeasure_h
#define __itkJacobianGradientSimilarityMeasure_h

#include "itkSimilarityMeasure.h"


namespace itk
{
/** 
 * \class JacobianGradientSimilarityMeasure
 * \brief AbstractBase class, just to implement a gradient method based on Jacobian.
 * 
 * This class is inspired by ITKs origin itkMeanSquaresImageToImageMetric, so I have
 * just generalized it using TemplateMethod.
 *
 * Note that this class is NOT thread safe.
 * 
 * \ingroup RegistrationMetrics
 */
template < typename TFixedImage, typename TMovingImage > 
class ITK_EXPORT JacobianGradientSimilarityMeasure : 
    public SimilarityMeasure< TFixedImage, TMovingImage>
{
public:

  /** Standard class typedefs. */
  typedef JacobianGradientSimilarityMeasure              Self;
  typedef SimilarityMeasure<TFixedImage, TMovingImage >  Superclass;
  typedef SmartPointer<Self>                             Pointer;
  typedef SmartPointer<const Self>                       ConstPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro(JacobianGradientSimilarityMeasure, SimilarityMeasure);

  /** Types transferred from the base class */
  typedef typename Superclass::MovingImageType            MovingImageType;
  typedef typename Superclass::FixedImageType             FixedImageType;
  typedef typename Superclass::FixedImageConstPointer     FixedImageConstPointer;
  typedef typename Superclass::GradientImageType          GradientImageType;
  typedef typename Superclass::GradientPixelType          GradientPixelType;
  typedef typename Superclass::DerivativeType             DerivativeType;
  typedef typename Superclass::InputPointType             InputPointType;
  typedef typename Superclass::OutputPointType            OutputPointType;
  typedef typename Superclass::RealType                   RealType;
  typedef typename Superclass::MeasureType                MeasureType;
  typedef typename Superclass::TransformParametersType    TransformParametersType;
  typedef typename Superclass::TransformJacobianType      TransformJacobianType;
  
  /** Get the derivatives of the match measure. */
  void GetCostFunctionDerivative( const TransformParametersType & parameters, DerivativeType  & derivative ) const;

protected:
  
  JacobianGradientSimilarityMeasure();
  virtual ~JacobianGradientSimilarityMeasure() {};
  void PrintSelf(std::ostream& os, Indent indent) const;

  /** 
   * For derived class to reset any internal variables.
   */
  virtual void ResetDerivativeComputations() {};
  
  /** 
   * For derived classes to do the necessary agregation of the derivative at each point.
   */
  virtual void ComputeDerivativeValue(
      DerivativeType  & derivative,               // The array of derivatives, one per parameter
      const GradientPixelType & gradientPixel,    // The gradient at the moving image pixel
      const TransformJacobianType & jacobianType, // The jacobian at that moving image pixel    
      unsigned int imageDimensions,               // The number of dimensions (ie. 2D or 3D)
      unsigned int parameterNumber,               // Which parameter are we operating on?
      RealType fixedValue,                        // The fixed image value
      RealType movingValue) { }                   // The moving image value
  
  /**
   * For derived classes to do any finalising before handing the array back.
   */
  virtual void FinalizeDerivative(DerivativeType  & derivative) {};
  
private:
  
  JacobianGradientSimilarityMeasure(const Self&); // purposefully not implemented
  void operator=(const Self&);                    // purposefully not implemented

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkJacobianGradientSimilarityMeasure.txx"
#endif

#endif



