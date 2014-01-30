/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkUltrasoundPointerCalibrationCostFunction_h
#define itkUltrasoundPointerCalibrationCostFunction_h

#include <itkUltrasoundCalibrationCostFunction.h>
#include <vtkMatrix4x4.h>

namespace itk {

/**
 * \class UltrasoundPointerCalibrationCostFunction
 * \brief Cost function to minimise the squared distance error of
 * transformed ultrasound points to points from a tracked pointer.
 *
 * The parameters array should be set before optimisation with a reasonable starting estimate
 * using the this->SetInitialPosition(parameters) method in the base class. This class
 * can optimise different numbers of degrees of freedom as follows:
 * <pre>
 * 6DOF: 6 rigid (rx, ry, rz in Rodrigues formulation, tx, ty, tz in millimetres).
 * 8DOF: 6 rigid + 2 scaling
 * </pre>
 * The order of parameters is important.
 */
class UltrasoundPointerCalibrationCostFunction : public itk::UltrasoundCalibrationCostFunction
{

public:

  typedef UltrasoundPointerCalibrationCostFunction Self;
  typedef itk::UltrasoundCalibrationCostFunction   Superclass;
  typedef itk::SmartPointer<Self>                  Pointer;
  typedef itk::SmartPointer<const Self>            ConstPointer;

  itkNewMacro( Self );

  typedef Superclass::ParametersType           ParametersType;
  typedef Superclass::DerivativeType           DerivativeType;
  typedef Superclass::MeasureType              MeasureType;

  /**
   * \brief Sets the pointer offset, which is the position
   * of the pointer tip, relative to the origin of the tracking
   * sensors placed on the pointer. This is normally the output
   * of a Pivot Calibration of some sort. i.e. the pointer calibration.
   */
  void SetPointerOffset(const mitk::Point3D& pointerOffset);

  /**
   * \brief Gets the pointer offset.
   */
  mitk::Point3D GetPointerOffset() const;

  /**
   * \brief Sets the matrix to transform from the coordinate system
   * that tracks the pointer to the coordinate system that tracks
   * the probe.
   */
  void SetPointerTrackerToProbeTrackerTransform(const vtkMatrix4x4& matrix);

  /**
   * \brief Sets the matrix to transform from the probe coordinate
   * system to the coordinate system of the tracker that tracks the probe.
   */
  void SetProbeToProbeTrackerTransform(const vtkMatrix4x4& matrix);

  /**
   * \brief The cost function is the residual error of the reconstructed points,
   * where this function returns an array of n (x, y, z) tuples where n is the number
   * of points, and each x, y, z measure is the difference from the tracked pointer points.
   */
  virtual MeasureType GetValue( const ParametersType & parameters ) const;

protected:

  UltrasoundPointerCalibrationCostFunction();
  virtual ~UltrasoundPointerCalibrationCostFunction();

  UltrasoundPointerCalibrationCostFunction(const UltrasoundPointerCalibrationCostFunction&); // Purposefully not implemented.
  UltrasoundPointerCalibrationCostFunction& operator=(const UltrasoundPointerCalibrationCostFunction&); // Purposefully not implemented.

private:

  cv::Matx44d   m_PointerTrackerToProbeTrackerTransform;
  cv::Matx44d   m_ProbeToProbeTrackerTransform;
  mitk::Point3D m_PointerOffset;
};

} // end namespace

#endif
