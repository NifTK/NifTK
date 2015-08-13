/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkUltrasoundPointerCalibrationCostFunction_h
#define niftkUltrasoundPointerCalibrationCostFunction_h

#include <itkMultipleValuedCostFunction.h>
#include <mitkPointSet.h>
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>

namespace niftk {

/**
* \class UltrasoundPointerCalibrationCostFunction
* \brief Cost function for Ultrasound Pointer based calibration, a la Muratore 2001.
*/
class UltrasoundPointerCalibrationCostFunction : public itk::MultipleValuedCostFunction
{

public:

  typedef UltrasoundPointerCalibrationCostFunction Self;
  typedef itk::MultipleValuedCostFunction          Superclass;
  typedef itk::SmartPointer<Self>                  Pointer;
  typedef itk::SmartPointer<const Self>            ConstPointer;

  itkNewMacro( Self );

  typedef Superclass::ParametersType               ParametersType;
  typedef Superclass::DerivativeType               DerivativeType;
  typedef Superclass::MeasureType                  MeasureType;

  /**
  * \brief Equal to the number of points * 3.
  */
  virtual unsigned int GetNumberOfValues(void) const;

  /**
  * \brief Required by base class to return the number of parameters.
  */
  virtual unsigned int GetNumberOfParameters() const;

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
  * \brief Returns the rigid matrix for the given parameters.
  */
  vtkSmartPointer<vtkMatrix4x4> GetRigidMatrix(const ParametersType & parameters) const;

  /**
  * \brief Returns the scaling matrix for the given parameters.
  */
  vtkSmartPointer<vtkMatrix4x4> GetScalingMatrix(const ParametersType & parameters) const;

  /**
  * \brief The cost function is the residual error of the reconstructed point,
  * where this function returns an array of n (x, y, z) tuples where n is the number
  * of points, and each x, y, z measure is the difference from the invariant point.
  */
  virtual MeasureType GetValue( const ParametersType & parameters ) const;

  /**
  * \brief Sets the points in image space.
  */
  void SetImagePoints(const mitk::PointSet::Pointer imagePoints);

  /**
  * \brief Sets the points in sensor space.
  */
  void SetSensorPoints(const mitk::PointSet::Pointer sensorPoints);

protected:

  UltrasoundPointerCalibrationCostFunction();
  virtual ~UltrasoundPointerCalibrationCostFunction();

  UltrasoundPointerCalibrationCostFunction(const UltrasoundPointerCalibrationCostFunction&); // Purposefully not impl.
  UltrasoundPointerCalibrationCostFunction& operator=(
      const UltrasoundPointerCalibrationCostFunction&); // Purposefully not impl.

  /**
  * \brief Checks the supplied parameters array is the right size
  * (i.e. it equals this->GetNumberOfParameters()), and throws mitk::Exception if it isnt.
  */
  void ValidateSizeOfParametersArray(const ParametersType & parameters) const;

  /**
  * \brief Checks the supplied parameters array is the right size
  * (i.e. it equals this->m_Scales.GetSize()), and throws mitk::Exception if it isnt.
  */
  void ValidateSizeOfScalesArray(const ParametersType & parameters) const;

private:

  unsigned int            m_NumberOfParameters;
  ParametersType          m_Scales;
  mitk::PointSet::Pointer m_ImagePoints;
  mitk::PointSet::Pointer m_SensorPoints;
};

} // end namespace

#endif
