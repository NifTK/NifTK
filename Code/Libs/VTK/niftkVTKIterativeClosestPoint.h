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
  * Perform a vtkIterative closest point registration on the two data sets
  */
  bool Run();

  vtkSmartPointer<vtkMatrix4x4> GetTransform();
  bool ApplyTransform(vtkPolyData * solution);
  void SetSource (vtkSmartPointer<vtkPolyData>);
  void SetTarget (vtkSmartPointer<vtkPolyData>);
  void SetMaxLandmarks ( int);
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
