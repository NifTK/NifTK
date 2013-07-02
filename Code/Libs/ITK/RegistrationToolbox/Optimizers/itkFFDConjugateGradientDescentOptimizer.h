/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef ITKFFDCONJUGATEGRADIENTDESCENTOPTIMIZER_H_
#define ITKFFDCONJUGATEGRADIENTDESCENTOPTIMIZER_H_


#include "itkFFDGradientDescentOptimizer.h"
#include <itkImageToImageMetricWithConstraint.h>
#include <itkBSplineTransform.h>
#include <itkRegistrationForceFilter.h>
#include <itkBSplineSmoothVectorFieldFilter.h>
#include <itkInterpolateVectorFieldFilter.h>
#include <itkScaleVectorFieldFilter.h>

namespace itk
{
  
/** 
 * \class FFDConjugateGradientDescentOptimizer
 * \brief Class to perform FFD specific optimization using conjugate gradient descent.
 *
 * \ingroup Numerics Optimizers
 */  
template <class TFixedImage, class TMovingImage, class TScalarType, class TDeformationScalar>
class ITK_EXPORT FFDConjugateGradientDescentOptimizer : 
    public FFDGradientDescentOptimizer<TFixedImage, TMovingImage, TScalarType, TDeformationScalar>
{
public:
  
  /** 
   * Standard class typedefs. 
   */
  typedef FFDConjugateGradientDescentOptimizer                         Self;
  typedef FFDGradientDescentOptimizer<TFixedImage, TMovingImage, 
                                      TScalarType, TDeformationScalar> Superclass;
  typedef SmartPointer<Self>                                           Pointer;
  typedef SmartPointer<const Self>                                     ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Standard Type Macro. */
  itkTypeMacro( FFDConjugateGradientDescentOptimizer, FFDGradientDescentOptimizer );

  /** Get the number of dimensions we are working in. */
  itkStaticConstMacro(Dimension, unsigned int, TFixedImage::ImageDimension);

  /** Typedefs. */
  typedef typename Superclass::SimilarityMeasureType                   SimilarityMeasureType;
  typedef typename SimilarityMeasureType::TransformParametersType      ParametersType;
  typedef typename Superclass::BSplineTransformPointer                 BSplineTransformPointer;
  typedef typename Superclass::GridImagePointer                        GridImagePointer;
  typedef typename Superclass::OutputImageSizeType                     OutputImageSizeType;
  
protected:
  
  FFDConjugateGradientDescentOptimizer(); 
  virtual ~FFDConjugateGradientDescentOptimizer() {};
  void PrintSelf(std::ostream& os, Indent indent) const;

  /** Calculate a potential step following the gradient direction. */
  virtual void OptimizeNextStep(int iterationNumber, int numberOfGridVoxels, const ParametersType& current, ParametersType& next);
  
private:

  FFDConjugateGradientDescentOptimizer(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  /** Set up arrays for conjugate gradient. */
  virtual void Initialize();
  
  /** Store derivative of cost function. */
  virtual void StoreGradient(const ParametersType& gradient);
  
  /** Generate next gradient */
  void GetNextGradient(const ParametersType& currentGradient, ParametersType& nextGradient);
  
  /** Clean-up arrays for conjugate gradient. */
  virtual void CleanUp();

  struct float3{
    float x,y,z;
  };  
  
  float3 *conjugateG;
  float3 *conjugateH;
  unsigned long int m_NumberOfGridVoxels;
  ParametersType m_DerivativeAtCurrentPosition;
  ParametersType m_DerivativeAtNextPosition;
};

} // namespace itk.

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkFFDConjugateGradientDescentOptimizer.txx"
#endif

#endif /*ITKFFDCONJUGATEGRADIENTDESCENTOPTIMIZER_H_*/



