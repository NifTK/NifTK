/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkSingleResolutionImageRegistrationMethod_h
#define itkSingleResolutionImageRegistrationMethod_h


#include <itkImageRegistrationMethod.h>
#include <itkIterationUpdateCommand.h>
#include <itkSimilarityMeasure.h>

namespace itk
{

/**
 * \class SingleResolutionImageRegistrationMethod
 * \brief Base class for NifTK Image Registration Methods
 *
 * This Class extends the ITK ImageRegistrationMethod, implementing TemplateMethod [2]
 * to set things up before registration, and then call DoRegistration(), which can
 * be overriden.
 *
 * \ingroup RegistrationFilters
 */
template <typename TFixedImage, typename TMovingImage>
class ITK_EXPORT SingleResolutionImageRegistrationMethod : public ImageRegistrationMethod<TFixedImage, TMovingImage> 
{
public:
  /** Standard class typedefs. */
  typedef SingleResolutionImageRegistrationMethod            Self;
  typedef ImageRegistrationMethod<TFixedImage, TMovingImage> Superclass;
  typedef SmartPointer<Self>                                 Pointer;
  typedef SmartPointer<const Self>                           ConstPointer;
  typedef itk::IterationUpdateCommand                        IterationUpdateCommandType;
  typedef IterationUpdateCommandType::Pointer                IterationUpdateCommandPointer;
  
  /** Method for creation through the object factory. */
  itkNewMacro(Self);
  
  /** Run-time type information (and related methods). */
  itkTypeMacro(SingleResolutionImageRegistrationMethod, ImageRegistrationMethod);

  /** Typedefs */
  typedef typename Superclass::ParametersType                      ParametersType;
  typedef typename Superclass::TransformType                       TransformType;
  typedef typename Superclass::InterpolatorType                    InterpolatorType;

  /** Set/Get the IterationUpdateCommand. */
  itkSetObjectMacro( IterationUpdateCommand, IterationUpdateCommandType );
  itkGetObjectMacro( IterationUpdateCommand, IterationUpdateCommandType );
  
  /**
   * Set/Get interpolators. 
   */
  itkSetObjectMacro(FixedImageInterpolator, InterpolatorType); 
  itkSetObjectMacro(MovingImageInterpolator, InterpolatorType); 
  itkGetObjectMacro(FixedImageInterpolator, InterpolatorType); 
  itkGetObjectMacro(MovingImageInterpolator, InterpolatorType); 
  
  /** 
   * Initialize by setting the interconnects between the components. 
   * Override to initialise the interpolators. 
   */
  virtual void Initialize() throw (ExceptionObject);

protected:
  SingleResolutionImageRegistrationMethod();
  virtual ~SingleResolutionImageRegistrationMethod() {};
  void PrintSelf(std::ostream& os, Indent indent) const;

  /** 
   * Method invoked by the pipeline in order to 
   * trigger the computation of the registration. 
   */
  void  GenerateData ();

  /** Override this method to actually do the registration. */
  virtual void DoRegistration() throw (ExceptionObject);
  
  /**
   * For symmetric registration, we need to interpolate fixed image and moving image. 
   */
  /**
   * Fixed image interpolator. 
   */
  typename InterpolatorType::Pointer m_FixedImageInterpolator; 
  /**
   * Moving image interpolator. 
   */
  typename InterpolatorType::Pointer m_MovingImageInterpolator; 
  
private:
  
  SingleResolutionImageRegistrationMethod(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
  
  /** To print out the registration params as we go. */
  IterationUpdateCommandPointer m_IterationUpdateCommand;
  
};


} // end namespace itk


#ifndef ITK_MANUAL_INSTANTIATION
#include "itkSingleResolutionImageRegistrationMethod.txx"
#endif

#endif



