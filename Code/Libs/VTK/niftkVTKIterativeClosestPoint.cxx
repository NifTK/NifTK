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
#include <vtkVersion.h>
#include <stdexcept>
#include <limits>

#define __NIFTTKVTKICPNPOINTS 50
#define __NIFTTKVTKICPMAXITERATIONS 100

namespace niftk
{

//-----------------------------------------------------------------------------
VTKIterativeClosestPoint::VTKIterativeClosestPoint()
: m_Source(NULL)
, m_Target(NULL)
, m_TransformMatrix(NULL)
{
  m_ICP = vtkSmartPointer<vtkIterativeClosestPointTransform>::New();
  m_ICP->GetLandmarkTransform()->SetModeToRigidBody();
  m_ICP->SetMaximumNumberOfLandmarks(__NIFTTKVTKICPNPOINTS);
  m_ICP->SetMaximumNumberOfIterations(__NIFTTKVTKICPMAXITERATIONS);
  m_ICP->SetCheckMeanDistance(true);

  m_TransformMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
  m_TransformMatrix->Identity();
}


//-----------------------------------------------------------------------------
VTKIterativeClosestPoint::~VTKIterativeClosestPoint()
{
}


//-----------------------------------------------------------------------------
void VTKIterativeClosestPoint::SetMaxLandmarks(int maxLandMarks)
{
  m_ICP->SetMaximumNumberOfLandmarks(maxLandMarks);
  //TODO I'm not sure this works, changing the number of landmarks
  //doesn't seem to alter the performance of the algorithm.
}


//-----------------------------------------------------------------------------
void VTKIterativeClosestPoint::SetMaxIterations(int maxIterations)
{
  m_ICP->SetMaximumNumberOfIterations(maxIterations);
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
double VTKIterativeClosestPoint::Run()
{
  double residual = std::numeric_limits<double>::max();

  if (m_Source == NULL)
  {
    throw std::runtime_error("VTKIterativeClosestPoint::Run, source is NULL.");
  }
  if (m_Source->GetNumberOfPoints() < 3)
  {
    throw std::runtime_error("VTKIterativeClosestPoint::Run, source has < 3 points.");
  }
  if (m_Target == NULL)
  {
    throw std::runtime_error("VTKIterativeClosestPoint::Run, target is NULL.");
  }
  if (m_Target->GetNumberOfPoints() < 3)
  {
    throw std::runtime_error("VTKIterativeClosestPoint::Run, target has < 3 points.");
  }

  // VTK ICP is point to surface
  //   the source only needs points,
  //   but the target needs a surface

  if ( m_Target->GetNumberOfCells() == 0 )
  {
    if ( m_Source->GetNumberOfCells() == 0 )
    {
      throw std::runtime_error("Neither source not target have a surface, cannot run ICP");
    }
    m_ICP->SetSource(m_Target);
    m_ICP->SetTarget(m_Source);
    m_ICP->Modified();
    m_ICP->Update();
    m_TransformMatrix->DeepCopy(m_ICP->GetMatrix());
    m_TransformMatrix->Invert();
  }
  else
  {
    m_ICP->SetSource(m_Source);
    m_ICP->SetTarget(m_Target);
    m_ICP->Modified();
    m_ICP->Update();
    m_TransformMatrix->DeepCopy(m_ICP->GetMatrix());
  }
  residual = m_ICP->GetMeanDistance();
  return residual;
}


//-----------------------------------------------------------------------------
vtkSmartPointer<vtkMatrix4x4> VTKIterativeClosestPoint::GetTransform()
{
  vtkSmartPointer<vtkMatrix4x4> result = vtkSmartPointer<vtkMatrix4x4>::New();
  result->Identity();
  result->DeepCopy(m_TransformMatrix);
  return result;
}


//-----------------------------------------------------------------------------
void VTKIterativeClosestPoint::ApplyTransform(vtkPolyData * solution)
{
  if (m_Source == NULL)
  {
    throw std::runtime_error("VTKIterativeClosestPoint::ApplyTransform, source is NULL.");
  }
  if (m_TransformMatrix == NULL)
  {
    throw std::runtime_error("VTKIterativeClosestPoint::ApplyTransform, transform matrix is NULL.");
  }
  if (solution == NULL)
  {
    throw std::runtime_error("VTKIterativeClosestPoint::ApplyTransform, solution vtkPolyData is NULL.");
  }

  vtkSmartPointer<vtkTransformPolyDataFilter> icpTransformFilter = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
  vtkSmartPointer<vtkTransform> icpTransform = vtkSmartPointer<vtkTransform>::New();
  icpTransform->SetMatrix(m_TransformMatrix);

  // Clear all memory.
  solution->Initialize();

#if VTK_MAJOR_VERSION <= 5
  icpTransformFilter->SetInput(m_Source);
#else
  icpTransformFilter->SetInputData(m_Source);
#endif
  icpTransformFilter->SetOutput(solution);
  icpTransformFilter->SetTransform(icpTransform);
  icpTransformFilter->Update();
}

//-----------------------------------------------------------------------------
} // end namespace
