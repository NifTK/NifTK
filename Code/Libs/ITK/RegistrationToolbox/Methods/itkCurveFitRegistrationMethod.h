/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkCurveFitRegistrationMethod_h
#define __itkCurveFitRegistrationMethod_h

#include "itkImageToImageFilter.h"

#include "itkImage.h"
#include "itkImageReconstructionMetric.h"
#include "itkMultipleValuedNonLinearOptimizer.h"
#include "itkBSplineCurveFitMetric.h"


namespace itk
{

/** \class CurveFitRegistrationMethod \brief Method for
 * registering a temporal volume by constraining the time-varying
 * intensities to be smooth as measured by a BSpline goodness of fit metric.
 *
 * The method uses a generic optimizer that can
 * be selected at run-time. The only restriction for the optimizer is
 * that it should be able to operate in single-valued cost functions
 * given that the metrics used to compare images provide a single 
 * value as output.
 */
template < class IntensityType = int >
class ITK_EXPORT CurveFitRegistrationMethod :
  public ImageToImageFilter< Image<IntensityType, 4>, Image<IntensityType, 4> > 
{
  public:
      
  /** Standard class typedefs. */
  typedef CurveFitRegistrationMethod                    Self;
  typedef ImageToImageFilter< Image<IntensityType, 4>, 
                              Image<IntensityType, 4> > Superclass;
  typedef SmartPointer<Self>                            Pointer;
  typedef SmartPointer<const Self>                      ConstPointer;
      
  /** Method for creation through the object factory. */
  itkNewMacro(Self);
      
  /** Run-time type information (and related methods). */
  itkTypeMacro(CurveFitRegistrationMethod, ImageToImageFilter);
      
  /// Some convenient typedefs.
  static const unsigned int NDimensions = 4;

  typedef Image<IntensityType, NDimensions>        TemporalVolumeType;
  typedef typename TemporalVolumeType::Pointer     TemporalVolumePointer;
  typedef typename TemporalVolumeType::RegionType  TemporalVolumeRegionType;
  typedef typename TemporalVolumeType::PixelType   TemporalVolumePixelType;
  typedef typename TemporalVolumeType::SizeType    TemporalVolumeSizeType;
  typedef typename TemporalVolumeType::SpacingType TemporalVolumeSpacingType;
  typedef typename TemporalVolumeType::PointType   TemporalVolumePointType;

   /// Type of the optimizer.
  typedef MultipleValuedNonLinearOptimizer  OptimizerType;
  typedef typename OptimizerType::Pointer   OptimizerPointer;
      
  /// The type of the metric
  typedef BSplineCurveFitMetric< IntensityType > MetricType;
  typedef typename MetricType::Pointer           MetricPointer;
 
  /**  Type of the Transform . */
  typedef typename MetricType::TransformType TransformType;
  typedef typename TransformType::Pointer    TransformPointer;
     
  /** Type of the optimisation (transformation) parameters - the
      images are transformed inside the metric. This is the type
      used to represent the search space of the optimization
      algorithm */
  typedef typename MetricType::TransformParametersType ParametersType;
      
  /** Set/Get the initial transformation parameters. */
  void SetInitialTransformParameters( const ParametersType &param );
  itkGetConstReferenceMacro( InitialParameters, ParametersType );

  /** Get the last transformation parameters visited by 
   * the optimizer. */
  itkGetConstReferenceMacro( LastParameters, ParametersType );
      
  /** Set/Get the Optimizer. */
  itkSetObjectMacro( Optimizer,  OptimizerType );
  itkGetObjectMacro( Optimizer,  OptimizerType );
      
  /** Set/Get the Metric. */
  itkSetObjectMacro( Metric, MetricType );
  itkGetObjectMacro( Metric, MetricType );

  /** Set/Get the Transfrom. */
  itkSetObjectMacro( Transform, TransformType );
  itkGetObjectMacro( Transform, TransformType );
   
  /** Method to return the latest modified time of this object or
   * any of its cached ivars */
  unsigned long GetMTime() const;  

  protected:

  CurveFitRegistrationMethod();
  virtual ~CurveFitRegistrationMethod() {};

  void PrintSelf(std::ostream& os, Indent indent) const;

  /** Initialise by setting the interconnects between the components. */
  virtual void Initialise() throw (ExceptionObject);
       
  /** Method that initiates the optimization process. This method should not be
   * called directly by the users. Instead, this method is intended to be
   * invoked internally by GenerateData() which is in turn invoked by
   * the Update() method. */
  void StartOptimization(void);

  /** Method invoked by the pipeline in order to trigger the registration. */
  void GenerateData();



  private:

  CurveFitRegistrationMethod(const Self&); // purposely not implemented
  void operator=(const Self&);	          // purposely not implemented
  
  bool m_FlagInitialised;

  TemporalVolumePointer m_InputTemporalVolume;

  OptimizerPointer      m_Optimizer;
  MetricPointer         m_Metric;

  TransformPointer      m_Transform;

  ParametersType        m_InitialParameters;
  ParametersType        m_LastParameters;

};


} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkCurveFitRegistrationMethod.txx"
#endif

#endif




