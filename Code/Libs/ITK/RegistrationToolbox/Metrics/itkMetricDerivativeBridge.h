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
#ifndef __itkMetricDerivativeBridge_h
#define __itkMetricDerivativeBridge_h

#include "itkObject.h"
#include "itkImageToImageMetricWithConstraint.h"

namespace itk
{

template <class TFixedImage, class TMovingImage> class ImageToImageMetricWithConstraint;

/** 
 * \class DerivativeBridge
 * \brief AbstractBase class, implementing Bridge [2] to provide an
 * interface for anything that calculates derivatives.
 * 
 * \ingroup RegistrationMetrics
 */
template < typename TFixedImage, typename TMovingImage > 
class ITK_EXPORT MetricDerivativeBridge : 
    public Object
{
public:

  /** Standard class typedefs. */
  typedef MetricDerivativeBridge                          Self;
  typedef Object                                          Superclass;
  typedef SmartPointer<Self>                              Pointer;
  typedef SmartPointer<const Self>                        ConstPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro(MetricDerivativeBridge, Object);

  /** Typedefs. */
  typedef ImageToImageMetricWithConstraint<TFixedImage, TMovingImage> SimilarityMeasureType;
  typedef typename SimilarityMeasureType::ConstPointer                SimilarityMeasurePointer;
  typedef typename SimilarityMeasureType::DerivativeType              DerivativeType;
  typedef typename SimilarityMeasureType::TransformParametersType     ParametersType;

  /** Get the number of dimensions we are working in. */
  itkStaticConstMacro(Dimension, unsigned int, TFixedImage::ImageDimension);

  /** Get the derivatives of the measure, which writes into the supplied DerivativeType. */
  virtual void GetCostFunctionDerivative(SimilarityMeasurePointer similarityMeasure, 
                     const ParametersType &parameters,
                     DerivativeType &derivative) const = 0;

protected:
  
  MetricDerivativeBridge() {};
  virtual ~MetricDerivativeBridge() {};

private:
  
  MetricDerivativeBridge(const Self&); // purposefully not implemented
  void operator=(const Self&);    // purposefully not implemented
};

} // end namespace itk

#endif



