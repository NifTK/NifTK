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
#include <vtkIterativeClosestPointTransform.h>
#include <vtkCellLocator.h>

namespace niftk {

/**
 * \class VTKIterativeClosestPoint
 * \brief Uses vtkIterativeClosestPointTransform to register two vtkPolyData sets.
 */
class NIFTKVTK_WINEXPORT VTKIterativeClosestPoint {

public:

  VTKIterativeClosestPoint();
  ~VTKIterativeClosestPoint();

  /**
   * \brief Perform a vtk Iterative Closest Point (ICP) registration on the two data sets.
   */
  void Run();

  /**
   * \brief returns the transform to move the source to the target.
   */
  vtkSmartPointer<vtkMatrix4x4> GetTransform() const;

  /**
   * \brief Returns the RMS residual error between target and transformed source.
   */
  double GetRMSResidual() const;

  /**
   * \brief Transform the source to the target, placing the result in solution.
   */
  void ApplyTransform(vtkPolyData * solution);

  /**
   * \brief Set the source poly data.
   */
  void SetSource (vtkSmartPointer<vtkPolyData>);

  /**
   * \brief Set the target polydata.
   */
  void SetTarget (vtkSmartPointer<vtkPolyData>);

  /**
   * \brief Set the maximum number of landmarks.
   */
  void SetMaxLandmarks(int);

  /**
   * \brief Set the maximum number of iterations.
   */
  void SetMaxIterations(int);

private:

  vtkSmartPointer<vtkIterativeClosestPointTransform>  m_ICP;
  vtkSmartPointer<vtkPolyData>                        m_Source;
  vtkSmartPointer<vtkPolyData>                        m_Target;
  vtkSmartPointer<vtkMatrix4x4>                       m_TransformMatrix;
  vtkSmartPointer<vtkCellLocator>                     m_Locator;
};

} // end namespace

#endif
