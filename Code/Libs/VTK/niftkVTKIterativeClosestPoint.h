/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __NIFTKVTKITERATIVECLOSESTPOINT_H
#define __NIFTKVTKITERATIVECLOSESTPOINT_H


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

namespace niftk
{

  class NIFTKVTK_WINEXPORT IterativeClosestPoint {

    public:
      IterativeClosestPoint();
      ~IterativeClosestPoint();
    /* \brief
    * Perform a vtkIterative closest point registration on the two data sets
    */
      bool Run();

      vtkSmartPointer<vtkMatrix4x4> GetTransform();
      bool TransformSource();
      bool TransformTarget();
      void SetSource (vtkSmartPointer<vtkPolyData>);
      void SetTarget (vtkSmartPointer<vtkPolyData>);
      void SetMaxLandmarks ( int);
      void SetMaxIterations(int);


    private:
      vtkSmartPointer<vtkIterativeClosestPointTransform> m_icp;
      vtkSmartPointer<vtkPolyData> m_Source;
      vtkSmartPointer<vtkPolyData> m_Target;
      vtkSmartPointer<vtkMatrix4x4> m_TransformMatrix;
      unsigned int m_MaxLandmarks;
      unsigned int m_MaxIterations;
  };



} // end namespace niftk

#endif  //__NIFTKVTKITERATIVECLOSESTPOINT_H
