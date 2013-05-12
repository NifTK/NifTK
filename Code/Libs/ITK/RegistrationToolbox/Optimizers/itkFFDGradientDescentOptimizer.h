/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef ITKFFDGRADIENTDESCENTOPTIMIZER_H_
#define ITKFFDGRADIENTDESCENTOPTIMIZER_H_

#include <NifTKConfigure.h>
#include <niftkITKWin32ExportHeader.h>
#include "itkLocalSimilarityMeasureGradientDescentOptimizer.h"
#include <itkImageToImageMetricWithConstraint.h>
#include <itkBSplineTransform.h>
#include <itkRegistrationForceFilter.h>
#include <itkBSplineSmoothVectorFieldFilter.h>
#include <itkInterpolateVectorFieldFilter.h>
#include <itkScaleVectorFieldFilter.h>
#include <itkScalarImageToNormalizedGradientVectorImageFilter.h>

namespace itk
{
  
/** 
 * \class FFDGradientDescentOptimizer
 * \brief Class to perform FFD specific optimization.
 *
 * \ingroup Numerics Optimizers
 */  
template <class TFixedImage, class TMovingImage, class TScalarType, class TDeformationScalar>
class ITK_EXPORT FFDGradientDescentOptimizer :
    public LocalSimilarityMeasureGradientDescentOptimizer<TFixedImage, TMovingImage, TScalarType, TDeformationScalar>
{
public:
  
  /** 
   * Standard class typedefs. 
   */
  typedef FFDGradientDescentOptimizer                                                     Self;
  typedef LocalSimilarityMeasureGradientDescentOptimizer<TFixedImage, TMovingImage, 
                                                         TScalarType, TDeformationScalar> Superclass;
  typedef SmartPointer<Self>                                                              Pointer;
  typedef SmartPointer<const Self>                                                        ConstPointer;

  /** Standard Type Macro. */
  itkTypeMacro( FFDGradientDescentOptimizer, LocalSimilarityMeasureGradientDescentOptimizer );

  /** Get the number of dimensions we are working in. */
  itkStaticConstMacro(Dimension, unsigned int, TFixedImage::ImageDimension);

  /** Typedefs. */
  typedef DeformableTransform<TFixedImage, TScalarType, 
                              Dimension, TDeformationScalar>               DeformableTransformType;
  typedef typename DeformableTransformType::Pointer                        DeformableTransformPointer;
  typedef BSplineTransform<TFixedImage, TScalarType, 
                           Dimension, TDeformationScalar>                  BSplineTransformType;
  typedef BSplineTransformType*                                            BSplineTransformPointer;
  
  typedef TScalarType                                                      MeasureType;
  typedef ImageToImageMetricWithConstraint<TFixedImage, TMovingImage>      SimilarityMeasureType;
  typedef typename SimilarityMeasureType::ConstPointer                     SimilarityMeasurePointer;
  typedef typename SimilarityMeasureType::TransformParametersType          ParametersType;
  typedef typename SimilarityMeasureType::DerivativeType                   DerivativeType;

  typedef typename BSplineTransformType::GridImageType                     GridImageType;
  typedef GridImageType*                                                   GridImagePointer;
  
  typedef Vector< TDeformationScalar, itkGetStaticConstMacro(Dimension) >  VectorPixelType;
  typedef Image< VectorPixelType, itkGetStaticConstMacro(Dimension) >      VectorImageType;

  /** FFD Pipeline as follows: First generate the force. */
  typedef RegistrationForceFilter<TFixedImage, TMovingImage, 
                                               TDeformationScalar>         ForceFilterType;
  typedef typename ForceFilterType::Pointer                                ForceFilterPointer;
  typedef typename ForceFilterType::OutputImageType                        ForceOutputImageType;
  
  /** FFD Pipeline as follows: Then Smooth it. */
  typedef BSplineSmoothVectorFieldFilter<TDeformationScalar, Dimension>    SmoothFilterType;
  typedef typename SmoothFilterType::Pointer                               SmoothFilterPointer;
  
  /** FFD Pipeline as follows: Then calculate force at grid points. */
  typedef InterpolateVectorFieldFilter<TDeformationScalar, Dimension>      InterpolateFilterType;
  typedef typename InterpolateFilterType::Pointer                          InterpolateFilterPointer;  

  /** Internal filters. */
  typedef ScaleVectorFieldFilter<TDeformationScalar, Dimension>            ScaleFieldType;
  typedef typename ScaleFieldType::Pointer                                 ScaleFieldPointer;
  typedef ScalarImageToNormalizedGradientVectorImageFilter<TFixedImage, TDeformationScalar> GradientFilterType;
  typedef typename GradientFilterType::Pointer                             GradientFilterPointer;
  
  /** The output of the interpolated vector field is...... Vectors! */
  typedef typename InterpolateFilterType::OutputImageType             OutputImageType;
  typedef typename OutputImageType::PixelType                         OutputImagePixelType;
  typedef typename OutputImageType::Pointer                           OutputImagePointer;
  typedef ImageRegionIterator<OutputImageType>                        OutputImageIteratorType;
  typedef typename OutputImageType::SizeType                          OutputImageSizeType;
  typedef typename OutputImageType::SpacingType                       OutputImageSpacingType;
  
  /** Set the force filter to use. */ 
  itkSetObjectMacro( ForceFilter, ForceFilterType );
  itkGetConstObjectMacro( ForceFilter, ForceFilterType );

  /** Set the smoothing filter to use. */
  itkSetObjectMacro( SmoothFilter, SmoothFilterType );
  itkGetConstObjectMacro( SmoothFilter, SmoothFilterType );
  
  /** Set the interpolation filter to use. */
  itkSetObjectMacro( InterpolatorFilter, InterpolateFilterType );
  itkGetConstObjectMacro( InterpolatorFilter, InterpolateFilterType );

  /** 
   * Set a tolerance, such that, if the maximum observed gradient vector magnitude
   * is below this threshold, we treat it as zero. When we come to multiply
   * by the step size, it stops us dividing by really small numbers.
   */
  itkSetMacro(MinimumGradientVectorMagnitudeThreshold, TScalarType);
  itkGetMacro(MinimumGradientVectorMagnitudeThreshold, TScalarType);
  
  /** If true, we multiply force vectors, by gradient image. Default false. */
  itkSetMacro(ScaleForceVectorsByGradientImage, bool);
  itkGetMacro(ScaleForceVectorsByGradientImage, bool);
  
  /** If true, when we multiply force vectors by gradient image, we
   * multiply each component, and if false, we multiply the force
   * vector by the magnitude of the gradient image vector. */
  itkSetMacro(ScaleByComponents, bool);
  itkGetMacro(ScaleByComponents, bool);
  
  /** If true, we apply BSpline smoothing of gradient vectors,
   * and if false we dont.  Defaults to true. */
  itkSetMacro(SmoothGradientVectorsBeforeInterpolatingToControlPointLevel, bool);
  itkGetMacro(SmoothGradientVectorsBeforeInterpolatingToControlPointLevel, bool);
  
  /** Turn off/on dumping the force image. Default off.*/
  itkSetMacro(WriteForceImage, bool);
  itkGetMacro(WriteForceImage, bool);
  
  /** Filename for force image. */
  itkSetMacro(ForceImageFileName, std::string);
  itkGetMacro(ForceImageFileName, std::string);

  /** File Extension for force image. */
  itkSetMacro(ForceImageFileExt, std::string);
  itkGetMacro(ForceImageFileExt, std::string);

protected:
  
  FFDGradientDescentOptimizer(); 
  virtual ~FFDGradientDescentOptimizer() {};
  void PrintSelf(std::ostream& os, Indent indent) const;

  /** Calculate a potential step following the gradient direction. */
  virtual double CalculateNextStep(int iterationNumber, double currentSimilarity, const ParametersType& current, ParametersType& next);

  /** Just get the gradient. */
  virtual void GetGradient(int iterationNumber, const ParametersType& current, ParametersType& next);

  /** Performs a line ascent. */
  virtual bool LineAscent(int iterationNumber, int numberOfGridVoxels, const ParametersType& current, ParametersType& next);
  
  /** Called by CalculateNextStep (which itself is called in base classes), so once we 
   * have a parameters array full of derivative vectors, subclasses can decide what to do with it. */
  virtual void OptimizeNextStep(int iterationNumber, int numberOfGridVoxels, const ParametersType& current, ParametersType& next) {};
  
  /** We inject a force filter. */
  ForceFilterPointer m_ForceFilter;
  
  /** We inject a smoothing filter. */
  SmoothFilterPointer m_SmoothFilter;
  
  /** We inject the interpolator filter. */
  InterpolateFilterPointer m_InterpolatorFilter;

  /** Internal filter, but available to subclasses. */
  ScaleFieldPointer m_ScaleVectorFieldFilter;
  
  /** Internak filter, but available to subclasses. */
  GradientFilterPointer m_GradientImageFilter;

  /** To count how many iterations we call CalculateNextStep */
  unsigned int m_CalculateNextStepCounter;

private:

  FFDGradientDescentOptimizer(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  /** So we can threshold noise out of the gradient vectors. */
  TScalarType m_MinimumGradientVectorMagnitudeThreshold;
  
  /** If requested, we multiply force vectors by gradient image. */
  bool m_ScaleForceVectorsByGradientImage;
  
  /** We can either multiply by gradient magnitude, or by each component. */
  bool m_ScaleByComponents;
  
  /** Nice name. */
  bool m_SmoothGradientVectorsBeforeInterpolatingToControlPointLevel;
  
  /** Flag to turn off/on dumping of force image at each iteration. */
  bool m_WriteForceImage;
  
  /** File name for force image. */
  std::string m_ForceImageFileName;
  
  /** File extension for deformation field. */
  std::string m_ForceImageFileExt;

};

} // namespace itk.

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkFFDGradientDescentOptimizer.txx"
#endif

#endif /*ITKFFDGRADIENTDESCENTOPTIMIZER_H_*/



