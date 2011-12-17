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
#ifndef __itkMultiResolutionImageRegistrationWrapper_h
#define __itkMultiResolutionImageRegistrationWrapper_h

#include "itkProcessObject.h"
#include "itkNumericTraits.h"
#include "itkDataObjectDecorator.h"
#include "itkImageToImageMetric.h"
#include "itkSingleValuedNonLinearOptimizer.h"
#include "itkRecursiveMultiResolutionPyramidImageFilter.h"
#include "itkMaskedImageRegistrationMethod.h"
#include "itkBinaryThresholdImageFilter.h"

namespace itk
{

/** 
 * \class MultiResolutionImageRegistrationWrapper
 * \brief UCL Base class for multi-resolution image registration methods.
 *
 * Here, we are providing a simple set of multi-resolution
 * image pyramids and then giving the data to a single resolution registration method.
 * 
 * \sa ImageRegistrationMethod
 * \ingroup RegistrationFilters
 */
template <typename TInputImageType, class TPyramidFilter = RecursiveMultiResolutionPyramidImageFilter< TInputImageType, TInputImageType > >
class ITK_EXPORT MultiResolutionImageRegistrationWrapper : public ProcessObject 
{
public:
  /** Standard class typedefs. */
  typedef MultiResolutionImageRegistrationWrapper  Self;
  typedef ProcessObject                            Superclass;
  typedef SmartPointer<Self>                       Pointer;
  typedef SmartPointer<const Self>                 ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);
  
  /** Run-time type information (and related methods). */
  itkTypeMacro(MultiResolutionImageRegistrationWrapper, ProcessObject);

  /**  Type of the Internal Single Resolution Registration Method. */
  typedef MaskedImageRegistrationMethod<TInputImageType>                            SingleResType;
  typedef typename SingleResType::Pointer                                           SingleResPointer;

  /**  Type of the input image. */
  typedef          TInputImageType                                                  InputImageType;
  typedef typename InputImageType::PixelType                                        InputImagePixelType;
  typedef typename InputImageType::Pointer                                          InputImagePointer;
  typedef typename InputImageType::ConstPointer                                     InputImageConstPointer;
  typedef typename InputImageType::RegionType                                       InputImageRegionType;

  /**  Type of the metric. */
  typedef ImageToImageMetric< InputImageType, InputImageType >                      MetricType;
  typedef typename MetricType::Pointer                                              MetricPointer;

  /**  Type of the Transform . */
  typedef typename MetricType::TransformType                                        TransformType;
  typedef typename TransformType::Pointer                                           TransformPointer;

  /** Type for the output: Using Decorator pattern for enabling
   *  the Transform to be passed in the data pipeline */
  typedef  DataObjectDecorator< TransformType >                                     TransformOutputType;
  typedef typename TransformOutputType::Pointer                                     TransformOutputPointer;
  typedef typename TransformOutputType::ConstPointer                                TransformOutputConstPointer;
  
  /**  Type of the Interpolator. */
  typedef typename MetricType::InterpolatorType                                     InterpolatorType;
  typedef typename InterpolatorType::Pointer                                        InterpolatorPointer;

  /**  Type of the optimizer. */
  typedef SingleValuedNonLinearOptimizer                                            OptimizerType;

  /** Threshold mask to 0 and 1, just in case its not binary already. */
  typedef BinaryThresholdImageFilter<InputImageType, InputImageType>                ThresholdFilterType;
  typedef typename ThresholdFilterType::Pointer                                     ThresholdFilterPointer;

  /** Type of the image multiresolution pyramid. */
  typedef TPyramidFilter                                                               ImagePyramidType;
  typedef typename ImagePyramidType::Pointer                                           ImagePyramidPointer;
  typedef typename ImagePyramidType::ScheduleType                                      ImagePyramidScheduleType;
  
  /** Type of the Transformation parameters This is the same type used to
   *  represent the search space of the optimization algorithm */
  typedef  typename MetricType::TransformParametersType ParametersType;

  /** Smart Pointer type to a DataObject. */
  typedef typename DataObject::Pointer DataObjectPointer;

  /** For the schedule */
  typedef typename ImagePyramidType::ScheduleType                                      ScheduleType;

  /** Method that initiates the registration. */
  virtual void StartRegistration();

  /** Method to stop the registration. */
  virtual void StopRegistration();

  /** Set/Get the SingleRes type, this is what actually does the registration. */
  itkSetObjectMacro( SingleResMethod, SingleResType );
  itkGetObjectMacro( SingleResMethod, SingleResType );
  
  /** Set/Get the Fixed image. */
  itkSetConstObjectMacro( FixedImage, InputImageType );
  itkGetConstObjectMacro( FixedImage, InputImageType ); 

  /** Set/Get the Fixed mask. */
  itkSetConstObjectMacro( FixedMask, InputImageType );
  itkGetConstObjectMacro( FixedMask, InputImageType ); 

  /** Set/Get the Moving image. */
  itkSetConstObjectMacro( MovingImage, InputImageType );
  itkGetConstObjectMacro( MovingImage, InputImageType );

  /** Set/Get the Moving mask. */
  itkSetConstObjectMacro( MovingMask, InputImageType );
  itkGetConstObjectMacro( MovingMask, InputImageType );

  /** Set/Get the number of multi-resolution levels, default = 1. */
  itkSetMacro( NumberOfLevels, unsigned int );
  itkGetMacro( NumberOfLevels, unsigned int );

  /** 
   * Set/Get the start level.
   */
  itkSetMacro ( StartLevel, unsigned int);
  itkGetMacro ( StartLevel, unsigned int);

  /** 
   * Set/Get the stop level.
   */
  itkSetMacro ( StopLevel, unsigned int);
  itkGetMacro ( StopLevel, unsigned int);

  /** Get the current resolution level being processed. */
  itkGetMacro( CurrentLevel, unsigned int );

  /** Set/Get the initial transformation parameters. */
  //itkSetMacro( InitialTransformParameters, ParametersType );
  //itkGetConstReferenceMacro( InitialTransformParameters, ParametersType );
  virtual void SetInitialTransformParameters(const ParametersType& parameters) 
  { 
    this->m_InitialTransformParametersOfNextLevel = parameters; 
  }
  virtual const ParametersType& GetInitialTransformParameters() const 
  {
    return this->m_InitialTransformParametersOfNextLevel; 
  }

  /** 
   * Set/Get the initial transformation parameters of the next resolution
   * level to be processed. The default is the last set of parameters of
   * the last resolution level. 
   */
  itkSetMacro( InitialTransformParametersOfNextLevel, ParametersType );
  itkGetConstReferenceMacro( InitialTransformParametersOfNextLevel, ParametersType );

  /** 
   * Get the last transformation parameters visited by 
   * the optimizer. 
   */
  //itkGetConstReferenceMacro( LastTransformParameters, ParametersType );  
  virtual const ParametersType& GetLastTransformParameters() const
  {
    return this->m_SingleResMethod->GetLastTransformParameters(); 
  }

  /** Returns the transform resulting from the registration process  */
  const TransformOutputType * GetOutput() const;

  /** Make a DataObject of the correct type to be used as the specified
   * output. */
  virtual DataObjectPointer MakeOutput(unsigned int idx);

  /** Method to return the latest modified time of this object or
   * any of its cached ivars */
  unsigned long GetMTime() const;  
  
  /** Set the multi-resolution pyramid schedule */
  ScheduleType* GetSchedule() { return m_Schedule;}
  void SetSchedule(ScheduleType* schedule) 
    { 
      this->m_Schedule = schedule;
      this->m_UserSpecifiedSchedule = true;
    } 

  /** 
   * Set/Get the MaskBeforePyramid flag. Default false.
   */
  itkSetMacro ( MaskBeforePyramid, bool);
  itkGetMacro ( MaskBeforePyramid, bool);
  
  itkSetMacro (UseOriginalImageAtFinalLevel, bool); 
  itkGetMacro (UseOriginalImageAtFinalLevel, bool); 

protected:
  MultiResolutionImageRegistrationWrapper();
  virtual ~MultiResolutionImageRegistrationWrapper() {};
  void PrintSelf(std::ostream& os, Indent indent) const;

  /** 
   * Method invoked by the pipeline in order to trigger the computation of the registration. 
   */
  void  GenerateData ();

  /**
   * This is only called once, before the registration starts, to set up all the pyramids.
   */
  virtual void PreparePyramids();
  
  /** 
   * Initialize by setting the interconnects between the components of this class.
   * This method is executed at every level of the pyramid with the values corresponding to this resolution.
   */
  virtual void Initialize() throw (ExceptionObject);

  /**
   * This is for subclasses to implement. It gets called just before the optimisation.
   * In here, you should do anything specific for your method, eg. Fluid/BSpline etc.
   */
  virtual void BeforeSingleResolutionRegistration() {};

  /**
   * This is for subclasses to implement. It gets called just after the optimisation.
   * In here, you should do anything specific for your method, eg. Fluid/BSpline etc.
   */
  virtual void AfterSingleResolutionRegistration() {};

  unsigned int                        m_NumberOfLevels;
  unsigned int                        m_StartLevel;  
  unsigned int                        m_StopLevel;
  unsigned int                        m_CurrentLevel;
  ParametersType                      m_InitialTransformParametersOfNextLevel;
  SingleResPointer                    m_SingleResMethod;
  
  /**
   * Use original image at the final (i.e. finest) level if set to true. 
   */
  bool                                m_UseOriginalImageAtFinalLevel; 
  bool                                m_UserSpecifiedSchedule;

  InputImageConstPointer              m_FixedImage;
  InputImageConstPointer              m_FixedMask;
  InputImageConstPointer              m_MovingImage;
  InputImageConstPointer              m_MovingMask;

  ImagePyramidPointer                 m_FixedImagePyramid;
  ImagePyramidPointer                 m_FixedMaskPyramid;  
  ImagePyramidPointer                 m_MovingImagePyramid;
  ImagePyramidPointer                 m_MovingMaskPyramid;
  
  ThresholdFilterPointer              m_FixedMaskThresholder;
  ThresholdFilterPointer              m_MovingMaskThresholder;
  
  bool                                m_Stop;
  
  ScheduleType*                       m_Schedule;
  
  bool                                m_MaskBeforePyramid;
  
private:
  MultiResolutionImageRegistrationWrapper(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};


} // end namespace itk


#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMultiResolutionImageRegistrationWrapper.txx"
#endif

#endif



