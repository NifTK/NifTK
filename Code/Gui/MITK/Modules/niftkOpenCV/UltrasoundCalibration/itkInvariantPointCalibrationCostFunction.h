/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkInvariantPointCalibrationCostFunction_h
#define itkInvariantPointCalibrationCostFunction_h

#include <itkMultipleValuedCostFunction.h>
#include <cv.h>
#include <mitkVector.h>
#include <mitkTimeStampsContainer.h>
#include <mitkTrackingAndTimeStampsContainer.h>

namespace itk {

/**
 * \class InvariantPointCalibrationCostFunction
 * \brief Base class for Ultrasound Pin/Cross-Wire calibration and Video Hand-Eye calibration cost functions.
 *
 * Ultrasound:
 *   - 1 DOF: temporal calibration
 *   - 6 DOF: 3 rotations, 3 translations
 *   - 9 DOF: 6 DOF + invariant point
 *   - 11 DOF: 9 DOF + scaling
 *   - 12 DOF: 11 DOF + temporal calibration
 *
 * Video:
 *   - 1 DOF: temporal calibration
 *   - 6 DOF: 3 rotations, 3 translations
 *   - 9 DOF: 6 DOF + invariant point
 *   - 10 DOF: 9 DOF + temporal calibration
 *
 * \see itk::UltrasoundPinCalibrationCostFunction
 * \see itk::VideoHandEyeCalibrationCostFunction
 */
class InvariantPointCalibrationCostFunction : public itk::MultipleValuedCostFunction
{

public:

  typedef InvariantPointCalibrationCostFunction Self;
  typedef itk::MultipleValuedCostFunction       Superclass;
  typedef itk::SmartPointer<Self>               Pointer;
  typedef itk::SmartPointer<const Self>         ConstPointer;

  typedef Superclass::ParametersType            ParametersType;
  typedef Superclass::DerivativeType            DerivativeType;
  typedef Superclass::MeasureType               MeasureType;
  typedef mitk::TimeStampsContainer::TimeStamp  TimeStampType;

  itkSetMacro(InvariantPoint, mitk::Point3D);
  itkGetConstMacro(InvariantPoint, mitk::Point3D);

  itkSetMacro(OptimiseInvariantPoint, bool);
  itkGetConstMacro(OptimiseInvariantPoint, bool);

  itkSetMacro(TimingLag, double);
  itkGetConstMacro(TimingLag, double);
  
  itkSetMacro(AllowableTimingError, TimeStampType);
  itkGetConstMacro(AllowableTimingError, TimeStampType);

  itkSetMacro(OptimiseTimingLag, bool);
  itkGetConstMacro(OptimiseTimingLag, bool);

  itkSetMacro(OptimiseRigidTransformation, bool);
  itkGetConstMacro(OptimiseRigidTransformation, bool);

  itkSetMacro(Verbose, bool);
  itkGetConstMacro(Verbose, bool);

  void SetRigidTransformation(const cv::Matx44d& rigidBodyTrans);
  cv::Matx44d GetRigidTransformation() const;

  std::vector<double> GetRigidTransformationParameters() const;
  void SetRigidTransformationParameters(const std::vector<double>& params);

  /**
   * \brief Equal to the number of points * 3.
   */
  virtual unsigned int GetNumberOfValues(void) const;

  /**
   * \brief Required by base class to return the number of parameters.
   * See introduction to this class.
   */
  virtual unsigned int GetNumberOfParameters() const;

  /**
   * \brief Sets the number of parameters being optimised.
   * This should be called before optimisation starts.
   */
  void SetNumberOfParameters(const int& numberOfParameters);

  /**
   * \brief Simply uses central differences to approximate the derivative for each of the parameters.
   * See also SetScales where you set the relative size of each parameter step size.
   */
  virtual void GetDerivative( const ParametersType & parameters, DerivativeType  & derivative ) const;

  /**
   * \brief Used when calculating derivative using central differences.
   */
  void SetScales(const ParametersType& scales);

  /**
   * \brief Returns the RMS residual of all the values stored in the values array.
   */
  double GetResidual(const MeasureType& values) const;

  /**
   * \brief Sets the tracking data onto this object.
   */
  void SetTrackingData(mitk::TrackingAndTimeStampsContainer* trackingData);

  /**
   * \brief Sets the point data onto this object, setting the number of values = pointData.size()*3.
   */
  void SetPointData(std::vector< std::pair<unsigned long long, cv::Point3d> >* pointData);

  /**
   * \brief The cost function is the residual error of the reconstructed point,
   * where this function returns an array of n (x, y, z) tuples where n is the number
   * of points, and each x, y, z measure is the difference from the invariant point.
   */
  virtual MeasureType GetValue( const ParametersType & parameters ) const;

protected:

  InvariantPointCalibrationCostFunction();
  virtual ~InvariantPointCalibrationCostFunction();

  InvariantPointCalibrationCostFunction(const InvariantPointCalibrationCostFunction&); // Purposefully not implemented.
  InvariantPointCalibrationCostFunction& operator=(const InvariantPointCalibrationCostFunction&); // Purposefully not implemented.

  /**
   * \brief Checks the supplied parameters array is the right size (i.e. it equals this->GetNumberOfParameters()), and throws mitk::Exception if it isnt.
   */
  void ValidateSizeOfParametersArray(const ParametersType & parameters) const;

  /**
   * \brief Checks the supplied parameters array is the right size (i.e. it equals this->m_Scales.GetSize()), and throws mitk::Exception if it isnt.
   */
  void ValidateSizeOfScalesArray(const ParametersType & parameters) const;

  /**
   * \brief Computes the calibration (image-to-probe) or (hand-eye) transformation from the current estimate of the parameters.
   *
   * ie. Derived class itk::UltrasoundPinCalibrationCostFunction can choose to add scaling, whereas
   * itk::VideoHandEyeCalibrationCostFunction can choose not to for examples.
   */
  virtual cv::Matx44d GetCalibrationTransformation(const ParametersType & parameters) const = 0;

  /**
   * \brief Computes the rigid body part (US: image-to-probe) or (Video: hand-eye) transformation from the current estimate of the parameters.
   *
   * This is always 6 DOF, so no scaling.
   */
  cv::Matx44d GetRigidTransformation(const ParametersType & parameters) const;

  /**
   * \brief Computes the translation transformation.
   *
   * This is always 3 DOF, pure translation, translating from tracker coordinates to the invariant point.
   */
  cv::Matx44d GetTranslationTransformation(const ParametersType & parameters) const;

  /**
   * \brief Extracts the lag parameter from the array of things being optimised.
   */
  double GetLag(const ParametersType & parameters) const;

  ParametersType                                        m_Scales;
  mitk::Point3D                                         m_InvariantPoint;
  bool                                                  m_OptimiseInvariantPoint;
  double                                                m_TimingLag; //in seconds
  bool                                                  m_OptimiseTimingLag;
  std::vector<double>                                   m_RigidTransformation;
  bool                                                  m_OptimiseRigidTransformation;
  mutable unsigned int                                  m_NumberOfValues;
  TimeStampType                                         m_AllowableTimingError;
  unsigned int                                          m_NumberOfParameters;
  std::vector< std::pair<TimeStampType, cv::Point3d> > *m_PointData;
  mitk::TrackingAndTimeStampsContainer                 *m_TrackingData;
  bool                                                  m_Verbose;
};

} // end namespace

#endif
