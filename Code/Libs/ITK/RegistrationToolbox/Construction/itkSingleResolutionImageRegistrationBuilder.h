/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkSingleResolutionImageRegistrationBuilder_h
#define __itkSingleResolutionImageRegistrationBuilder_h


#include "itkObject.h"
#include "itkSingleResolutionImageRegistrationMethod.h"
#include "itkImageRegistrationFactory.h"

namespace itk
{

/** 
 * \class SingleResolutionImageRegistrationBuilder
 * \brief Base class for NifTK Image Registration Builders.
 *
 * The purpose of a Builder [2], as opposed to a Factory is that
 * it encapsulates the logic of creating a complex process in parts.
 * So to use this class, you call these methods in sequence:
 * 
 * StartCreation()
 * 
 * CreateInterpolator()
 * 
 * CreateMetric()
 * 
 * CreateTransform()
 * 
 * CreateOptimizer()
 * 
 * GetSingleResolutionImageRegistrationMethod()
 * 
 * So, any logic to do with which objects can be used with which other
 * objects, or in other words "inter object dependencies" go in here.
 * 
 * \sa ImageRegistrationFactory
 */
template <typename TImage, unsigned int Dimension, class TScalarType>
class ITK_EXPORT SingleResolutionImageRegistrationBuilder : public Object
{
  public:

    /** Standard class typedefs. */
    typedef SingleResolutionImageRegistrationBuilder Self;
    typedef Object Superclass;
    typedef SmartPointer<Self> Pointer;
    typedef SmartPointer<const Self> ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self);

    /** Run-time type information (and related methods). */
    itkTypeMacro(SingleResolutionImageRegistrationBuilder, Object);

    /** Typedefs. */
    typedef itk::ImageRegistrationFactory<TImage, Dimension, TScalarType>       ImageRegistrationFactoryType;
    typedef typename ImageRegistrationFactoryType::SingleResRegistrationType    SingleResRegType;
    typedef typename ImageRegistrationFactoryType::InterpolatorType             InterpolatorType;
    typedef typename ImageRegistrationFactoryType::TransformType                TransformType;
    typedef typename ImageRegistrationFactoryType::EulerAffineTransformType     EulerAffineTransformType;
    typedef EulerAffineTransformType*                                           EulerAffineTransformPointer;
    typedef typename ImageRegistrationFactoryType::MetricType                   MetricType;
    typedef typename MetricType::ConstPointer                                   MetricTypeConstPointer;
    typedef typename ImageRegistrationFactoryType::OptimizerType                OptimizerType;
    typedef typename ImageRegistrationFactoryType::GradientDescentType          GradientDescentType;
    typedef GradientDescentType*                                                GradientDescentPointer;

    /**  Type of the Fixed image. */
    typedef          TImage                             ImageType;
    typedef typename ImageType::ConstPointer            ImageConstPointer;
    typedef typename ImageType::RegionType              ImageRegionType;
    typedef typename ImageType::SizeType                ImageSizeType;
    typedef typename itk::Point<TScalarType, Dimension> InputPointType;
    
    /** Call this first. */
    typename SingleResRegType::Pointer StartCreation(SingleResRegistrationMethodTypeEnum type);

    /** Then create the interpolator. */
    typename InterpolatorType::Pointer CreateInterpolator(InterpolationTypeEnum type);

    /** Then create the metric. */
    typename MetricType::Pointer CreateMetric(MetricTypeEnum type);

    /** Then create the transform. We pass the image in, to guide initialization. */
    typename TransformType::Pointer CreateTransform(TransformTypeEnum type, ImageConstPointer image);
    
    /** Then create the transform. We pass the image in, to guide initialization, and read the initial parameters from a file */
    typename TransformType::Pointer CreateTransform(std::string initialTransformName);

    /** Then create the optimizer. We pass the image in, to guide initialization. */
    typename OptimizerType::Pointer CreateOptimizer(OptimizerTypeEnum type);

    /** Then retrieve the full object. */
    typename SingleResRegType::Pointer GetSingleResolutionImageRegistrationMethod();
    
    /** Then create the fixed image interpolator. */
    typename InterpolatorType::Pointer CreateFixedImageInterpolator(InterpolationTypeEnum type);

    /** Then create the moving image interpolator. */
    typename InterpolatorType::Pointer CreateMovingImageInterpolator(InterpolationTypeEnum type);

  protected:

    SingleResolutionImageRegistrationBuilder();
    virtual ~SingleResolutionImageRegistrationBuilder(){};
    void PrintSelf(std::ostream& os, Indent indent) const;

  private:

    SingleResolutionImageRegistrationBuilder(const Self&); //purposely not implemented
    void operator=(const Self&); //purposely not implemented

    /** Store this, between calls. */
    typename SingleResRegType::Pointer m_ImageRegistrationMethod;

    /** Reference to the factory, to help us build stuff. */
    typename ImageRegistrationFactoryType::Pointer m_ImageRegistrationFactory;
    
    /** Just so we can keep track of what we have created. */
    SingleResRegistrationMethodTypeEnum m_ImageRegistrationMethodEnum;
    
    /** Just so we can keep track of what we have created. */
    InterpolationTypeEnum m_InterpolatorEnum;
    
    /** Just so we can keep track of what we have created. */
    MetricTypeEnum m_MetricEnum;
    
    /** Just so we can keep track of what we have created. */
    TransformTypeEnum m_TransformEnum;
    
    /** Just so we can keep track of what we have created. */
    OptimizerTypeEnum m_OptimizerEnum;
    
  };

}
// end namespace itk


#ifndef ITK_MANUAL_INSTANTIATION
#include "itkSingleResolutionImageRegistrationBuilder.txx"
#endif

#endif

