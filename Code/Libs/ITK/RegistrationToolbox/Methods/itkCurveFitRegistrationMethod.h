/*=============================================================================

  NifTK: An image processing toolkit jointly developed by the
  Dementia Research Centre, and the Centre For Medical Image Computing
  at University College London.
 
  See:        http://dementia.ion.ucl.ac.uk/
  http://cmic.cs.ucl.ac.uk/
  http://www.ucl.ac.uk/

  Last Changed      : $Date: 2012-10-31 16:15:38 +0000 (Wed, 31 Oct 2012) $
  Revision          : $Revision: 9614 $
  Last modified by  : $Author: jhh $

  Original author   : j.hipwell@ucl.ac.uk

  Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notices for more information.

  ============================================================================*/

#ifndef __itkCurveFitRegistrationMethod_h
#define __itkCurveFitRegistrationMethod_h

#include "itkProcessObject.h"
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
class ITK_EXPORT CurveFitRegistrationMethod : public ProcessObject 
{
  public:
      
  /** Standard class typedefs. */
  typedef CurveFitRegistrationMethod  Self;
  typedef ProcessObject  Superclass;
  typedef SmartPointer<Self>   Pointer;
  typedef SmartPointer<const Self>  ConstPointer;
      
  /** Method for creation through the object factory. */
  itkNewMacro(Self);
      
  /** Run-time type information (and related methods). */
  itkTypeMacro(CurveFitRegistrationMethod, ProcessObject);
      
  /// Some convenient typedefs.
  static const unsigned int NDimensions = 4;

  typedef Image<IntensityType, NDimensions>             InputTemporalVolumeType;
  typedef typename InputTemporalVolumeType::Pointer     InputTemporalVolumePointer;
  typedef typename InputTemporalVolumeType::RegionType  InputTemporalVolumeRegionType;
  typedef typename InputTemporalVolumeType::PixelType   InputTemporalVolumePixelType;
  typedef typename InputTemporalVolumeType::SizeType    InputTemporalVolumeSizeType;
  typedef typename InputTemporalVolumeType::SpacingType InputTemporalVolumeSpacingType;
  typedef typename InputTemporalVolumeType::PointType   InputTemporalVolumePointType;

   /// Type of the optimizer.
  typedef MultipleValuedNonLinearOptimizer  OptimizerType;
  typedef typename OptimizerType::Pointer   OptimizerPointer;
      
  /// The type of the metric
  typedef BSplineCurveFitMetric< IntensityType > MetricType;
  typedef typename MetricType::Pointer           MetricPointer;
 
  /**  Type of the Transform . */
  typedef typename MetricType::TransformType TransformType;
  typedef typename TransformType::Pointer    TransformPointer;
  typedef std::vector< TransformPointer >    TransformListType;
     
  /** Type of the optimisation (transformation) parameters - the
      images are transformed inside the metric. This is the type
      used to represent the search space of the optimization
      algorithm */
  typedef typename MetricType::TransformParametersType ParametersType;
      
  /** Set/Get the initial transformation parameters. */
  virtual void SetInitialTransformParameters( const ParametersType & param );
  itkGetConstReferenceMacro( InitialTransformParameters, ParametersType );

  /** Get the last transformation parameters visited by 
   * the optimizer. */
  itkGetConstReferenceMacro( LastTransformParameters, ParametersType );

  /** Set/Get the input temporal volume. */
  itkSetObjectMacro( InputTemporalVolume,  InputTemporalVolumeType );
  itkGetObjectMacro( InputTemporalVolume,  InputTemporalVolumeType );
      
  /** Set/Get the Optimizer. */
  itkSetObjectMacro( Optimizer,  OptimizerType );
  itkGetObjectMacro( Optimizer,  OptimizerType );
      
  /** Set/Get the Metric. */
  itkSetObjectMacro( Metric, MetricType );
  itkGetObjectMacro( Metric, MetricType );

  /** Set/Get the Transfrom. */
  itkSetObjectMacro( TransformList, TransformListType );
  itkGetObjectMacro( TransformList, TransformListType );

  /** Initialise by setting the interconnects between the components. */
  virtual void Initialise() throw (ExceptionObject);

  /** Returns the transform resulting from the registration process  */
  const TransformOutputType * GetOutput() const;

  /** Make a DataObject of the correct type to be used as the specified
   * output. */
  virtual DataObjectPointer MakeOutput(unsigned int idx);

  /** Method to return the latest modified time of this object or
   * any of its cached ivars */
  unsigned long GetMTime() const;  
    

  protected:

  CurveFitRegistrationMethod();
  virtual ~CurveFitRegistrationMethod() {};

  void PrintSelf(std::ostream& os, Indent indent) const;

  /** We avoid propagating the input region to the output by
      overloading this function */
  virtual void GenerateOutputInformation() {};
  
  /** Method that initiates the optimization process. This method should not be
   * called directly by the users. Instead, this method is intended to be
   * invoked internally by the Update() method. */
  void StartOptimization(void);
      
  /** Method invoked by the pipeline in order to trigger the computation of 
   * the reconstruction. */
  void  GenerateData();


  private:
  CurveFitRegistrationMethod(const Self&); // purposely not implemented
  void operator=(const Self&);	          // purposely not implemented
  
  bool m_FlagInitialised;

  InputTemporalVolumePointer       m_InputTemporalVolume;

  OptimizerPointer                 m_Optimizer;
  MetricPointer                    m_Metric;

  TransformListType                m_Transform;

  ParametersType                   m_InitialParameters;
  ParametersType                   m_LastParameters;

};


} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkCurveFitRegistrationMethod.txx"
#endif

#endif




