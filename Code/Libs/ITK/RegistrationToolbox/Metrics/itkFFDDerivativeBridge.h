/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkFFDDerivativeBridge_h
#define __itkFFDDerivativeBridge_h


#include "itkMetricDerivativeBridge.h"
#include <itkRegistrationForceFilter.h>
#include <itkBSplineSmoothVectorFieldFilter.h>
#include <itkInterpolateVectorFieldFilter.h>
#include <itkBSplineTransform.h>

namespace itk
{
/** 
 * \class FFDDerivativeBridge
 * \brief FFDDerivative bridge to enable plugging a whole pipeline into a similarity measure
 * to measure the derivative of a cost function. 
 * 
 * \ingroup RegistrationMetrics
 */
template < typename TFixedImage, typename TMovingImage > 
class ITK_EXPORT FFDDerivativeBridge : 
    public MetricDerivativeBridge<TFixedImage, TMovingImage>
{
public:

  /** Standard class typedefs. */
  typedef FFDDerivativeBridge                                Self;
  typedef MetricDerivativeBridge<TFixedImage, TMovingImage>  Superclass;
  typedef SmartPointer<Self>                                 Pointer;
  typedef SmartPointer<const Self>                           ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(FFDDerivativeBridge, MetricDerivativeBridge);

  /** Get the number of dimensions we are working in. */
  itkStaticConstMacro(Dimension, unsigned int, TFixedImage::ImageDimension);
  
  /** Typedefs. */
  typedef typename Superclass::SimilarityMeasureType         SimilarityMeasureType;
  typedef typename Superclass::SimilarityMeasurePointer      SimilarityMeasurePointer;
  typedef typename Superclass::DerivativeType                DerivativeType;
  typedef typename Superclass::ParametersType                ParametersType;
  typedef BSplineTransform<TFixedImage, double, Dimension>   BSplineTransformType;
  typedef typename BSplineTransformType::Pointer             BSplineTransformPointer;
  typedef typename BSplineTransformType::GridImageType       GridImageType;
  typedef typename GridImageType::Pointer                    GridImagePointer;
  
  /** FFD Pipeline as follows: First generate the force. */
  typedef RegistrationForceFilter<TFixedImage, TMovingImage> ForceFilterType;
  typedef typename ForceFilterType::Pointer                  ForceFilterPointer;
  
  /** FFD Pipeline as follows: Then Smooth it. */
  typedef BSplineSmoothVectorFieldFilter<double, Dimension>  SmoothFilterType;
  typedef typename SmoothFilterType::Pointer                 SmoothFilterPointer;
  
  /** FFD Pipeline as follows: Then calculate force at grid points. */
  typedef InterpolateVectorFieldFilter<double, Dimension>    InterpolateFilterType;
  typedef typename InterpolateFilterType::Pointer            InterpolateFilterPointer;  
  typedef typename InterpolateFilterType::OutputImageType    OutputImageType;
  typedef typename OutputImageType::PixelType                OutputImagePixelType;
  typedef typename OutputImageType::Pointer                  OutputImagePointer;
  typedef ImageRegionIterator<OutputImageType>               OutputImageIteratorType;
  typedef typename OutputImageType::SizeType                 OutputImageSizeType;
  
  /** Get the derivatives of the measure, which writes into the supplied DerivativeType. */
  void GetCostFunctionDerivative(SimilarityMeasurePointer similarityMeasure,
                                         const ParametersType &parameters,
                                         DerivativeType &derivative) const;

  /** Set the grid to use. */ 
  itkSetObjectMacro( Grid, GridImageType );
  itkGetConstObjectMacro( Grid, GridImageType );

  /** Set the force filter to use. */ 
  itkSetObjectMacro( ForceFilter, ForceFilterType );
  itkGetConstObjectMacro( ForceFilter, ForceFilterType );

  /** Set the smoothing filter to use. */
  itkSetObjectMacro( SmoothFilter, SmoothFilterType );
  itkGetConstObjectMacro( SmoothFilter, SmoothFilterType );
  
  /** Set the interpolation filter to use. */
  itkSetObjectMacro( InterpolatorFilter, InterpolateFilterType );
  itkGetConstObjectMacro( InterpolatorFilter, InterpolateFilterType );
  
protected:
  
  FFDDerivativeBridge();
  virtual ~FFDDerivativeBridge() {};

  /** Pointer to control point grid. */
  GridImagePointer m_Grid;
  
  /** We inject a force filter. */
  ForceFilterPointer m_ForceFilter;
  
  /** We inject a smoothing filter. */
  SmoothFilterPointer m_SmoothFilter;
  
  /** We inject the interpolator filter. */
  InterpolateFilterPointer m_InterpolatorFilter;
  
private:
  
  FFDDerivativeBridge(const Self&); // purposefully not implemented
  void operator=(const Self&);    // purposefully not implemented
  void PrintSelf(std::ostream& os, Indent indent) const;

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkFFDDerivativeBridge.txx"
#endif

#endif



