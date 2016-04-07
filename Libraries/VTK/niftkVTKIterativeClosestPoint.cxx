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
#include <vtkIterativeClosestPointTransform.h>
#include <vtkTransform.h>
#include <vtkVersion.h>
#include <stdexcept>
#include <map>

namespace niftk
{

//-----------------------------------------------------------------------------
VTKIterativeClosestPoint::VTKIterativeClosestPoint()
: m_Source(NULL)
, m_Target(NULL)
, m_TransformMatrix(NULL)
, m_Locator(NULL)
, m_ICPMaxLandmarks(50)
, m_ICPMaxIterations(100)
, m_TLSPercentage(50)
, m_TLSIterations(0)
{
  m_TransformMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
  m_TransformMatrix->Identity();

  m_Locator = vtkSmartPointer<vtkCellLocator>::New();
}


//-----------------------------------------------------------------------------
VTKIterativeClosestPoint::~VTKIterativeClosestPoint()
{
}


//-----------------------------------------------------------------------------
void VTKIterativeClosestPoint::SetICPMaxLandmarks(unsigned int maxLandMarks)
{
  if (maxLandMarks < 3)
  {
    throw std::runtime_error("SetICPMaxLandmarks: maxLandMarks must be >= 3.");
  }
  m_ICPMaxLandmarks = maxLandMarks;
}


//-----------------------------------------------------------------------------
void VTKIterativeClosestPoint::SetICPMaxIterations(unsigned int maxIterations)
{
  if (maxIterations < 1)
  {
    throw std::runtime_error("SetICPMaxIterations: maxIterations must be >= 1.");
  }
  m_ICPMaxIterations = maxIterations;
}


//-----------------------------------------------------------------------------
void VTKIterativeClosestPoint::SetTLSPercentage(unsigned int percentage)
{
  if (percentage > 100)
  {
    throw std::runtime_error("SetTLSPercentage: percentage must be <= 100.");
  }
  if (percentage == 0)
  {
    throw std::runtime_error("SetTLSPercentage: percentage must be >= 1.");
  }
  m_TLSPercentage = percentage;
}


//-----------------------------------------------------------------------------
void VTKIterativeClosestPoint::SetTLSIterations(unsigned int iterations)
{
  m_TLSIterations = iterations;
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
vtkSmartPointer<vtkMatrix4x4> VTKIterativeClosestPoint::GetTransform() const
{
  vtkSmartPointer<vtkMatrix4x4> result = vtkSmartPointer<vtkMatrix4x4>::New();
  result->DeepCopy(m_TransformMatrix);
  return result;
}


//-----------------------------------------------------------------------------
bool VTKIterativeClosestPoint::CheckInverted(vtkPolyData *source, vtkPolyData* target) const
{
  bool inverted = false;

  // VTK ICP is point to surface
  //   the source only needs points,
  //   but the target needs a surface

  if ( target->GetNumberOfCells() == 0 )
  {
    if ( source->GetNumberOfCells() == 0 )
    {
      throw std::runtime_error("Neither source not target have a surface, cannot run ICP");
    }
    inverted = true;
  }
  return inverted;
}


//-----------------------------------------------------------------------------
int VTKIterativeClosestPoint::GetStepSize(vtkPolyData *source) const
{
  int stepSize = 1;

  vtkIdType numberSourcePoints = source->GetNumberOfPoints();
  if (numberSourcePoints > m_ICPMaxLandmarks)
  {
    stepSize = numberSourcePoints / m_ICPMaxLandmarks;
  }
  return stepSize;
}


//-----------------------------------------------------------------------------
vtkSmartPointer<vtkMatrix4x4> VTKIterativeClosestPoint::InternalRunICP(vtkPolyData *source,
                                                                       vtkPolyData* target,
                                                                       unsigned int landmarks,
                                                                       unsigned int iterations,
                                                                       bool inverted) const
{
  if (source == NULL)
  {
    throw std::runtime_error("VTKIterativeClosestPoint::InternalRunICP, source is NULL.");
  }
  if (source->GetNumberOfPoints() < 3)
  {
    throw std::runtime_error("VTKIterativeClosestPoint::InternalRunICP, source has < 3 points.");
  }
  if (target == NULL)
  {
    throw std::runtime_error("VTKIterativeClosestPoint::InternalRunICP, target is NULL.");
  }
  if (target->GetNumberOfPoints() < 3)
  {
    throw std::runtime_error("VTKIterativeClosestPoint::InternalRunICP, target has < 3 points.");
  }

  vtkSmartPointer<vtkIterativeClosestPointTransform> icp
      = vtkSmartPointer<vtkIterativeClosestPointTransform>::New();

  icp->GetLandmarkTransform()->SetModeToRigidBody();
  icp->SetMaximumNumberOfLandmarks(landmarks);
  icp->SetMaximumNumberOfIterations(iterations);
  icp->SetLocator(m_Locator);           // This is to avoid vtkIterativeClosestPointTransform creating a new one at each iteration of TLS.
//  icp->CheckMeanDistanceOn();         // These just stop the ICP early, if its not moving far each iteration.
//  icp->SetMaximumMeanDistance(0.001); // These just stop the ICP early, if its not moving far each iteration.
  icp->SetSource(source);
  icp->SetTarget(target);
  icp->Modified();
  icp->Update();

  vtkSmartPointer<vtkMatrix4x4> result = vtkSmartPointer<vtkMatrix4x4>::New();
  result->DeepCopy(icp->GetMatrix());

  if (inverted)
  {
    result->Invert();
  }
  return result;
}


//-----------------------------------------------------------------------------
double VTKIterativeClosestPoint::Run()
{
  vtkPolyData *source = m_Source;
  vtkPolyData *target = m_Target;

  bool inverted = this->CheckInverted(source, target);
  if (inverted)
  {
    source = m_Target;
    target = m_Source;
  }

  vtkSmartPointer<vtkMatrix4x4> result = vtkSmartPointer<vtkMatrix4x4>::New();

  // At this point, we know the Target has cells.
  m_Locator = vtkSmartPointer<vtkCellLocator>::New(); // always over-writes current locator.
  m_Locator->SetDataSet(target);
  m_Locator->SetNumberOfCellsPerBucket(1);
  m_Locator->BuildLocator();

  if (m_TLSIterations == 0)
  {
    // Normal ICP, no TLS.
    result = this->InternalRunICP(source, target, m_ICPMaxLandmarks, m_ICPMaxIterations, inverted);
  }
  else
  {
    vtkSmartPointer<vtkPoints> points[2];
    vtkSmartPointer<vtkPolyData> polies[2];

    for (int i = 0; i < 2; i++)
    {
      points[i] = vtkSmartPointer<vtkPoints>::New();
      polies[i] = vtkSmartPointer<vtkPolyData>::New();
    }

    int current = 0;
    int other = 1;
    double sourcePoint[4];

    // First fill temporary dataset with a subsampled set of points.
    int step = this->GetStepSize(source);
    vtkIdType numberSourcePoints = source->GetNumberOfPoints();
    vtkIdType numberOfPointsInserted = 0;
    for (vtkIdType pointCounter = 0; pointCounter < numberSourcePoints
         && numberOfPointsInserted < m_ICPMaxLandmarks; pointCounter += step)
    {
      source->GetPoint(pointCounter, sourcePoint); // this retrieves x, y, z.
      points[current]->InsertPoint(numberOfPointsInserted++, sourcePoint[0], sourcePoint[1], sourcePoint[2]);
    }
    polies[current]->SetPoints(points[current]);

    // Do a certain number of iterations of TLS based ICP.
    for (int i = 0; i < m_TLSIterations; i++)
    {
      result = this->InternalRunICP(polies[current], target, points[current]->GetNumberOfPoints(), m_ICPMaxIterations, inverted);

      // Now iterate through all points, and form sorted list of residual errors.
      double transformedSourcePoint[4];
      double closestTargetPoint[4];
      vtkIdType cellId = 0;
      int subId = 0;
      double distance = 0;
      std::map<double, vtkIdType> map;

      // Get residual for each point.
      for (vtkIdType pointCounter = 0; pointCounter < points[current]->GetNumberOfPoints(); pointCounter++)
      {
        points[current]->GetPoint(pointCounter, sourcePoint); // this retrieves x, y, z.
        sourcePoint[3] = 1;                                   // but we need the w (homogeneous coords) for matrix multiply.

        result->MultiplyPoint(sourcePoint, transformedSourcePoint);
        m_Locator->FindClosestPoint(transformedSourcePoint,
                                    closestTargetPoint,
                                    cellId,
                                    subId,
                                    distance);
        std::pair<double, vtkIdType> pair(distance, pointCounter);
        map.insert(pair);
      }

      std::map<double, vtkIdType>::size_type numberOfPointsCopied = 0;
      std::map<double, vtkIdType>::size_type numberOfPointsInMap = map.size();
      std::map<double, vtkIdType>::size_type numberOfPointsRequired = numberOfPointsInMap
          * (static_cast<double>(m_TLSPercentage)/100.0);

      // Iterate through the top m_TLSPercentage of closest points, and put into other point set
      points[other]->Initialize();
      std::map<double, vtkIdType>::iterator iter = map.begin();
      do
      {
        points[current]->GetPoint(iter->second, sourcePoint);
        points[other]->InsertPoint(numberOfPointsCopied++, sourcePoint[0], sourcePoint[1], sourcePoint[2]);
        iter++;
      } while (iter != map.end() && numberOfPointsCopied < numberOfPointsRequired);

      polies[other]->SetPoints(points[other]);

      // Swap current/other
      int tmp = current;
      current = other;
      other = tmp;
    }

  }

  // Finish, set the member variable.
  m_TransformMatrix->DeepCopy(result);
  double residual = this->InternalGetRMSResidual(*source, *m_Locator, *m_TransformMatrix);
  return residual;
}


//-----------------------------------------------------------------------------
double VTKIterativeClosestPoint::GetRMSResidual(vtkPolyData& source) const
{
  return this->InternalGetRMSResidual(source, *m_Locator, *m_TransformMatrix);
}


//-----------------------------------------------------------------------------
double VTKIterativeClosestPoint::InternalGetRMSResidual(vtkPolyData& source,
                                                        vtkCellLocator& locator,
                                                        vtkMatrix4x4& matrix) const
{
  double residual = 0;
  double sourcePoint[4];
  double transformedSourcePoint[4];
  double closestTargetPoint[4];
  vtkIdType cellId;
  vtkIdType numberOfPointsUsed = 0;
  int subId;
  double distance;

  int step = this->GetStepSize(&source);
  vtkIdType numberSourcePoints = source.GetNumberOfPoints();
  for (vtkIdType pointCounter = 0; pointCounter < numberSourcePoints
       && numberOfPointsUsed < m_ICPMaxLandmarks; pointCounter+=step)
  {
    source.GetPoint(pointCounter, sourcePoint); // this retrieves x, y, z.
    sourcePoint[3] = 1;                         // but we need the w (homogeneous coords) for matrix multiply.

    matrix.MultiplyPoint(sourcePoint, transformedSourcePoint);
    locator.FindClosestPoint(transformedSourcePoint,
                             closestTargetPoint,
                             cellId,
                             subId,
                             distance);

    numberOfPointsUsed++;

    residual += (distance*distance);
/*
    std::cerr << "Matt, c=" << pointCounter
              << ", sp=" << sourcePoint[0]
              << ", " << sourcePoint[1]
              << ", " << sourcePoint[2]
              << ", tr=" << transformedSourcePoint[0]
              << ", " << transformedSourcePoint[1]
              << ", " << transformedSourcePoint[2]
              << ", tp" << closestTargetPoint[0]
              << ", " << closestTargetPoint[1]
              << ", " << closestTargetPoint[2]
              << ", dist=" << distance
              << ", resi=" << residual
              << std::endl;
*/
  }
  if (numberOfPointsUsed > 0)
  {
    residual /= static_cast<double>(numberOfPointsUsed);
  }
  residual = sqrt(residual);
  return residual;
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

  // Clear all memory.
  solution->Initialize();

  vtkSmartPointer<vtkTransform> icpTransform = vtkSmartPointer<vtkTransform>::New();
  icpTransform->SetMatrix(m_TransformMatrix);

  vtkSmartPointer<vtkTransformPolyDataFilter> icpTransformFilter = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
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
