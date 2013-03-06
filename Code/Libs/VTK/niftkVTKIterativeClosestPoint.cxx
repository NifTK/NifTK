/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkVTKIterativeClosestPoint.h"

#include <vtkLandmarkTransform.h>
#include <vtkPolyData.h>
#include <vtkIterativeClosestPointTransform.h>
#include <vtkTransformPolyDataFilter.h>

namespace niftk
{

 

// ---------------------------------------------------------------------------
// Create a reference image corresponding to a given control point grid image
// ---------------------------------------------------------------------------

  IterativeClosestPoint::IterativeClosestPoint()
    : m_Source(NULL),
    m_Target(NULL),
    m_TransformMatrix(NULL),
    m_MaxLandmarks(50),
    m_MaxIterations(100)
  {
    m_icp = vtkSmartPointer<vtkIterativeClosestPointTransform>::New();
    m_icp->GetLandmarkTransform()->SetModeToRigidBody();
    m_icp->SetMaximumNumberOfLandmarks(m_MaxLandmarks);
    m_icp->SetMaximumNumberOfIterations(m_MaxIterations);
  }

  IterativeClosestPoint::~IterativeClosestPoint()
  {}
  void IterativeClosestPoint::SetSource ( vtkSmartPointer<vtkPolyData>  source)
  {
    m_Source = source;
  }
  void IterativeClosestPoint::SetTarget ( vtkSmartPointer<vtkPolyData>  target)
  {
    m_Target = target;
  } 
  bool IterativeClosestPoint::Run()
  {
   if ( m_Source == NULL || m_Target == NULL ) 
   {
     return false;
   }
   m_icp->SetSource(m_Source);
   m_icp->SetTarget(m_Target);
   m_icp->Modified();
   m_icp->Update();
   m_TransformMatrix = m_icp->GetMatrix();
   return true;
  }

  vtkSmartPointer<vtkMatrix4x4> IterativeClosestPoint::GetTransform()
  {
    return m_TransformMatrix;
  }

  bool IterativeClosestPoint::TransformSource()
  {
   if ( m_Source == NULL && m_TransformMatrix == NULL ) 
   {
    return false;
   }
    vtkSmartPointer<vtkTransformPolyDataFilter> icpTransformFilter =
          vtkSmartPointer<vtkTransformPolyDataFilter>::New();
#if VTK_MAJOR_VERSION <= 5
    icpTransformFilter->SetInput(m_Source);
#else
    icpTransformFilter->SetInputData(m_Source);
#endif
    icpTransformFilter->SetTransform(m_icp);
    icpTransformFilter->Update();
    return true;
  }

  bool IterativeClosestPoint::TransformTarget()
  {
    return true;
  }



} // end namespace niftk

