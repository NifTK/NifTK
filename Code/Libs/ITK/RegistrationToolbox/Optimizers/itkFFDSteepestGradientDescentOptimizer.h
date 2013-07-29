/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkFFDSteepestGradientDescentOptimizer_h
#define itkFFDSteepestGradientDescentOptimizer_h

#include <NifTKConfigure.h>
#include <niftkITKWin32ExportHeader.h>
#include "itkFFDGradientDescentOptimizer.h"
#include <itkImageToImageMetricWithConstraint.h>
#include <itkUCLBSplineTransform.h>
#include <itkRegistrationForceFilter.h>
#include <itkBSplineSmoothVectorFieldFilter.h>
#include <itkInterpolateVectorFieldFilter.h>
#include <itkScaleVectorFieldFilter.h>

namespace itk
{
  
/** 
 * \class FFDSteepestGradientDescentOptimizer
 * \brief Class to perform FFD specific optimization using steepest gradient descent.
 *
 * \ingroup Numerics Optimizers
 */  
template <class TFixedImage, class TMovingImage, class TScalarType, class TDeformationScalar>
class ITK_EXPORT FFDSteepestGradientDescentOptimizer :
    public FFDGradientDescentOptimizer<TFixedImage, TMovingImage, TScalarType, TDeformationScalar>
{
public:
  
  /** 
   * Standard class typedefs. 
   */
  typedef FFDSteepestGradientDescentOptimizer                          Self;
  typedef FFDGradientDescentOptimizer<TFixedImage, TMovingImage, 
                                      TScalarType, TDeformationScalar> Superclass;
  typedef SmartPointer<Self>                                           Pointer;
  typedef SmartPointer<const Self>                                     ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Standard Type Macro. */
  itkTypeMacro( FFDSteepestGradientDescentOptimizer, FFDGradientDescentOptimizer );

  /** Get the number of dimensions we are working in. */
  itkStaticConstMacro(Dimension, unsigned int, TFixedImage::ImageDimension);

  /** Typedefs. */
  typedef typename Superclass::SimilarityMeasureType                   SimilarityMeasureType;
  typedef typename SimilarityMeasureType::TransformParametersType      ParametersType;

protected:
  
  FFDSteepestGradientDescentOptimizer(); 
  virtual ~FFDSteepestGradientDescentOptimizer() {};
  void PrintSelf(std::ostream& os, Indent indent) const;

  /** Calculate a potential step following the gradient direction. */
  virtual void OptimizeNextStep(int iterationNumber, int numberOfGridVoxels, const ParametersType& current, ParametersType& next);
  
private:

  FFDSteepestGradientDescentOptimizer(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};

} // namespace itk.

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkFFDSteepestGradientDescentOptimizer.txx"
#endif

#endif /*ITKFFDSTEEPESTGRADIENTDESCENTOPTIMIZER_H_*/



