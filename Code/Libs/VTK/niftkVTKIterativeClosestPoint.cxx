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
#include <vtkTransform.h>

namespace niftk
{

//-----------------------------------------------------------------------------
VTKIterativeClosestPoint::VTKIterativeClosestPoint()
  : m_Source(NULL),
  m_Target(NULL),
  m_TransformMatrix(NULL),
  m_MaxLandmarks(50),
  m_MaxIterations(100)
{
  m_Icp = vtkSmartPointer<vtkIterativeClosestPointTransform>::New();
  m_Icp->GetLandmarkTransform()->SetModeToRigidBody();
  m_Icp->SetMaximumNumberOfLandmarks(m_MaxLandmarks);
  m_Icp->SetMaximumNumberOfIterations(m_MaxIterations);
}


//-----------------------------------------------------------------------------
VTKIterativeClosestPoint::~VTKIterativeClosestPoint()
{
}


//-----------------------------------------------------------------------------
void VTKIterativeClosestPoint::SetMaxLandmarks(int MaxLandMarks)
{
  m_MaxLandmarks = MaxLandMarks;
  m_Icp->SetMaximumNumberOfLandmarks(m_MaxLandmarks);
  //TODO I'm not sure this works, changing the number of landmarks
  //doesn't seem to alter the performance of the algorithm.
}


//-----------------------------------------------------------------------------
void VTKIterativeClosestPoint::SetMaxIterations(int MaxIterations)
{
  m_MaxIterations = MaxIterations;
  m_Icp->SetMaximumNumberOfLandmarks(m_MaxIterations);
}


//-----------------------------------------------------------------------------
void VTKIterativeClosestPoint::SetSource ( vtkSmartPointer<vtkPolyData>  source)
{
  m_Source = source;
}


//-----------------------------------------------------------------------------
void VTKIterativeClosestPoint::SetTarget ( vtkSmartPointer<vtkPolyData>  target)
{
  m_Target = target;
}


//-----------------------------------------------------------------------------
bool VTKIterativeClosestPoint::Run()
{
  if ( m_Source == NULL || m_Target == NULL )
  {
    return false;
  }
  // VTK ICP is point to surface, the source only needs points,
  // but the target needs a surface

  if ( m_Target->GetNumberOfCells() == 0 )
  {
    if ( m_Source->GetNumberOfCells() == 0 )
    {
      std::cerr << "Neither source not target have a surface, cannot run ICP";
      return false;
    }
    m_Icp->SetSource(m_Target);
    m_Icp->SetTarget(m_Source);
    m_Icp->Modified();
    m_Icp->Update();
    m_TransformMatrix = m_Icp->GetMatrix();
    m_TransformMatrix->Invert();
  }
  else
  {
    m_Icp->SetSource(m_Source);
    m_Icp->SetTarget(m_Target);
    m_Icp->Modified();
    m_Icp->Update();
    m_TransformMatrix = m_Icp->GetMatrix();
  }
  return true;
}


//-----------------------------------------------------------------------------
vtkSmartPointer<vtkMatrix4x4> VTKIterativeClosestPoint::GetTransform()
{
  return m_TransformMatrix;
}


//-----------------------------------------------------------------------------
bool VTKIterativeClosestPoint::ApplyTransform(vtkPolyData * solution)
{
  if ( m_Source == NULL && m_TransformMatrix == NULL )
  {
    return false;
  }
  vtkSmartPointer<vtkTransformPolyDataFilter> icpTransformFilter = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
  vtkSmartPointer<vtkTransform> icpTransform = vtkSmartPointer<vtkTransform>::New();
  icpTransform->SetMatrix(m_TransformMatrix);

#if VTK_MAJOR_VERSION <= 5
  icpTransformFilter->SetInput(m_Source);
  icpTransformFilter->SetOutput(solution);
#else
  icpTransformFilter->SetInputData(m_Source);
  icpTransformFilter->SetOutputData(solution);
#endif
  icpTransformFilter->SetTransform(icpTransform);
  icpTransformFilter->Update();
  return true;
}

//-----------------------------------------------------------------------------
} // end namespace
