/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkInvNormalizedCorrelationImageToImageMetric_h
#define __itkInvNormalizedCorrelationImageToImageMetric_h

#include "itkImageToImageMetric.h"
#include "itkCovariantVector.h"
#include "itkPoint.h"


namespace itk
{
/** \class NormalizedCorrelationImageToImageMetric
 * \brief Computes similarity between two images to be registered
 *
 * This metric computes the correlation between pixels in the fixed image
 * and pixels in the moving image. The spatial correspondance between 
 * fixed and moving image is established through a Transform. Pixel values are
 * taken from the fixed image, their positions are mapped to the moving
 * image and result in general in non-grid position on it. Values at these
 * non-grid position of the moving image are interpolated using a user-selected
 * Interpolator. The correlation is normalized by the autocorrelations of both
 * the fixed and moving images.
 *
 * \ingroup RegistrationMetrics
 */
template < class TFixedImage, class TMovingImage > 
class ITK_EXPORT InvNormalizedCorrelationImageToImageMetric : 
    public ImageToImageMetric< TFixedImage, TMovingImage>
{
public:

  /** Standard class typedefs. */
  typedef InvNormalizedCorrelationImageToImageMetric         Self;
  typedef ImageToImageMetric<TFixedImage, TMovingImage >  Superclass;
  typedef SmartPointer<Self>                              Pointer;
  typedef SmartPointer<const Self>                        ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);
 
  /** Run-time type information (and related methods). */
  itkTypeMacro(InvNormalizedCorrelationImageToImageMetric, Object);

 
  /** Types transferred from the base class */
  typedef typename Superclass::RealType                 RealType;
  typedef typename Superclass::TransformType            TransformType;
  typedef typename Superclass::TransformPointer         TransformPointer;
  typedef typename Superclass::TransformParametersType  TransformParametersType;
  typedef typename Superclass::TransformJacobianType    TransformJacobianType;
  typedef typename Superclass::GradientPixelType        GradientPixelType;
  typedef typename Superclass::OutputPointType          OutputPointType;
  typedef typename Superclass::InputPointType           InputPointType;

  typedef typename Superclass::MeasureType              MeasureType;
  typedef typename Superclass::DerivativeType           DerivativeType;
  typedef typename Superclass::FixedImageType           FixedImageType;
  typedef typename Superclass::MovingImageType          MovingImageType;
  typedef typename Superclass::FixedImageConstPointer   FixedImageConstPointer;
  typedef typename Superclass::MovingImageConstPointer  MovingImageConstPointer;


  /** Get the derivatives of the match measure. */
  void GetDerivative( const TransformParametersType & parameters,
                      DerivativeType & Derivative ) const;

  /**  Get the value for single valued optimizers. */
  MeasureType GetValue( const TransformParametersType & parameters ) const;

  /**  Get value and derivatives for multiple valued optimizers. */
  void GetValueAndDerivative( const TransformParametersType & parameters,
                              MeasureType& Value, DerivativeType& Derivative ) const;

  /** Set/Get SubtractMean boolean. If true, the sample mean is subtracted 
   * from the sample values in the cross-correlation formula and
   * typically results in narrower valleys in the cost fucntion.
   * Default value is false. */
  itkSetMacro( SubtractMean, bool );
  itkGetConstReferenceMacro( SubtractMean, bool );
  itkBooleanMacro( SubtractMean );

protected:
  InvNormalizedCorrelationImageToImageMetric();
  virtual ~InvNormalizedCorrelationImageToImageMetric() {};
  void PrintSelf(std::ostream& os, Indent indent) const;

private:
  InvNormalizedCorrelationImageToImageMetric(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  bool    m_SubtractMean;

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkInvNormalizedCorrelationImageToImageMetric.txx"
#endif

#endif
