/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkUltrasoundPinCalibrationCostFunction_h
#define itkUltrasoundPinCalibrationCostFunction_h

#include <itkUltrasoundCalibrationCostFunction.h>

namespace itk {

/**
 * \class UltrasoundPinCalibrationCostFunction
 * \brief Cost function to minimise the squared distance error of
 * a cloud of points from the invariant point (normally 0,0,0).
 *
 * The parameters array should be set before optimisation with a reasonable starting estimate
 * using the this->SetInitialPosition(parameters) method in the base class. This class
 * can optimise different numbers of degrees of freedom as follows:
 * <pre>
 * 6DOF: 6 rigid (rx, ry, rz in Rodrigues formulation, tx, ty, tz in millimetres).
 * 8DOF: 6 rigid + 2 scaling (mm/pix)
 * 9DOF: 6 rigid + 3 invariant point (x, y, z location in millimetres).
 * 11DOF: 6 rigid + 2 scaling + 3 invariant point.
 * </pre>
 * The order of parameters is important.
 */
class UltrasoundPinCalibrationCostFunction : public itk::UltrasoundCalibrationCostFunction
{

public:

  typedef UltrasoundPinCalibrationCostFunction   Self;
  typedef itk::UltrasoundCalibrationCostFunction Superclass;
  typedef itk::SmartPointer<Self>                Pointer;
  typedef itk::SmartPointer<const Self>          ConstPointer;

  itkNewMacro( Self );

  typedef Superclass::ParametersType           ParametersType;
  typedef Superclass::DerivativeType           DerivativeType;
  typedef Superclass::MeasureType              MeasureType;

  /**
   * \brief Sets the number of invariant points that this class can store and optimise.
   *
   * This method must be called before any calls to SetInvariantPoint.
   */
  void SetNumberOfInvariantPoints(const unsigned int& numberOfInvariantPoints);

  /**
   * \brief Returns the number of invariant points.
   */
  unsigned int GetNumberOfInvariantPoints() const;

  /**
   * \brief Sets the invariant point specified by the parameter pointNumber.
   */
  void SetInvariantPoint(const unsigned int& pointNumber, const mitk::Point3D& invariantPoint);

  /**
   * \brief Gets the invariant point specified by the parameter pointNumber.
   */
  mitk::Point3D GetInvariantPoint(const unsigned int& pointNumber) const;

  /**
   * \brief The cost function is the residual error of the reconstructed point,
   * where this function returns an array of n (x, y, z) tuples where n is the number
   * of points, and each x, y, z measure is the difference from the invariant point.
   */
  virtual MeasureType GetValue( const ParametersType & parameters ) const;

protected:

  UltrasoundPinCalibrationCostFunction();
  virtual ~UltrasoundPinCalibrationCostFunction();

  UltrasoundPinCalibrationCostFunction(const UltrasoundPinCalibrationCostFunction&); // Purposefully not implemented.
  UltrasoundPinCalibrationCostFunction& operator=(const UltrasoundPinCalibrationCostFunction&); // Purposefully not implemented.

private:

  int                                m_NumberOfInvariantPoints;
  std::vector<mitk::Point3D>         m_InvariantPoints;


};

} // end namespace

#endif
