/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkVTKIterativeClosestPoint_h
#define niftkVTKIterativeClosestPoint_h

#include "niftkVTKWin32ExportHeader.h"
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>
#include <vtkCellLocator.h>

namespace niftk {

/**
 * \class VTKIterativeClosestPoint
 * \brief Uses vtkIterativeClosestPointTransform to register two vtkPolyData sets.
 *
 * This class requires that one (normally target) contains cells. i.e. a surface of
 * triangles for instance. If the source contains cells, but the target does not,
 * the registration is reversed, and then once completed, the transform is inverted.
 *
 * This class also implements a Trimmed Least Squares (TLS) approach,
 * whereby the ICP is repeated. At each iteration, controlled by SetTLSIterations(),
 * the best matching number of points, controlled by SetTLSPercentage() is retained,
 * and the outliers are discarded.
 */
class NIFTKVTK_WINEXPORT VTKIterativeClosestPoint {

public:

  VTKIterativeClosestPoint();
  ~VTKIterativeClosestPoint();

  /**
   * \brief Perform a vtk Iterative Closest Point (ICP) registration on the two data sets.
   * \return the RMS residual error, using the full source dataset.
   */
  double Run();

  /**
   * \brief Calculates the RMS residual, using the current transformation and the supplied source data-set.
   */
  double GetRMSResidual(vtkPolyData &source) const;

  /**
   * \brief returns the transform to move the source to the target.
   */
  vtkSmartPointer<vtkMatrix4x4> GetTransform() const;

  /**
   * \brief Transform the source to the target, placing the result in solution.
   */
  void ApplyTransform(vtkPolyData *solution);

  /**
   * \brief Set the source poly data.
   */
  void SetSource(vtkSmartPointer<vtkPolyData>);

  /**
   * \brief Set the target polydata.
   */
  void SetTarget(vtkSmartPointer<vtkPolyData>);

  /**
   * \brief Set the maximum number of ICP landmarks, default 50.
   */
  void SetICPMaxLandmarks(unsigned int);

  /**
   * \brief Set the maximum number of ICP iterations, default 100.
   */
  void SetICPMaxIterations(unsigned int);

  /**
   * \brief Set the number of TLS iterations, default 0.
   *
   * If zero, (the default), this feature is off.
   */
  void SetTLSIterations(unsigned int);

  /**
   * \brief Set the TLS percentage [1 - 100], default 50.
   */
  void SetTLSPercentage(unsigned int);

private:

  vtkSmartPointer<vtkPolyData>    m_Source;
  vtkSmartPointer<vtkPolyData>    m_Target;
  vtkSmartPointer<vtkMatrix4x4>   m_TransformMatrix;
  vtkSmartPointer<vtkCellLocator> m_Locator;
  unsigned int                    m_ICPMaxLandmarks;
  unsigned int                    m_ICPMaxIterations;
  unsigned int                    m_TLSPercentage;
  unsigned int                    m_TLSIterations;

  int GetStepSize(vtkPolyData *source) const;

  bool CheckInverted(vtkPolyData *source, vtkPolyData *target) const;

  vtkSmartPointer<vtkMatrix4x4> InternalRunICP(vtkPolyData *source,
                                               vtkPolyData *target,
                                               unsigned int landmarks,
                                               unsigned int iterations,
                                               bool inverted
                                              ) const;

  double InternalGetRMSResidual(vtkPolyData &source,
                                vtkCellLocator &locator,
                                vtkMatrix4x4 &matrix
                                ) const;
};

} // end namespace

#endif
