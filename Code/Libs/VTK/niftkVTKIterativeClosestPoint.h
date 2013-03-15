/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __niftkVTKIterativeClosestPoint_h
#define __niftkVTKIterativeClosestPoint_h


#include "NifTKConfigure.h"
#include "niftkCommonWin32ExportHeader.h"
#include "niftkVTKWin32ExportHeader.h"

#include <ostream>
#include <stdio.h>
#include <string>


#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>
#include <vtkIterativeClosestPointTransform.h>

#define __NIFTTKVTKICPNPOINTS 50
#define __NIFTTKVTKICPMAXITERATIONS 100

class NIFTKVTK_WINEXPORT niftkVTKIterativeClosestPoint {

public:
  niftkVTKIterativeClosestPoint();
  ~niftkVTKIterativeClosestPoint();
  /* \brief
  * Perform a vtk Iterative closest point registration on the two data sets
  */
  bool Run();

  /* 
   * \brief returns the transform to move the source to the target
   */
  vtkSmartPointer<vtkMatrix4x4> GetTransform();
  /* 
   * \brief Transform the source to the target, placing the result in solution
   */
  bool ApplyTransform(vtkPolyData * solution);
  /* 
   * \brief Set the source poly data
   */
  void SetSource (vtkSmartPointer<vtkPolyData>);
  /*
   * \brief Set the target polydata
   */
  void SetTarget (vtkSmartPointer<vtkPolyData>);
  /*
   * \brief Set the maximum number of landmarks, NOT WORKING
   */
  void SetMaxLandmarks ( int);
  /* 
   * \brief Set the maximum number of iterations.
   */
  void SetMaxIterations(int);

private:
  vtkSmartPointer<vtkIterativeClosestPointTransform>  m_Icp;
  vtkSmartPointer<vtkPolyData>                        m_Source;
  vtkSmartPointer<vtkPolyData>                        m_Target;
  vtkSmartPointer<vtkMatrix4x4>                       m_TransformMatrix;
  unsigned int                                        m_MaxLandmarks;
  unsigned int                                        m_MaxIterations;
};

#endif  //__NniftkVTKIterativeClosestPoint_h
