/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkICPBasedRegistration.h"
#include <niftkVTKIterativeClosestPoint.h>
#include <niftkPolyDataUtils.h>
#include <mitkExceptionMacro.h>
#include <limits>

namespace niftk
{

//-----------------------------------------------------------------------------
ICPBasedRegistration::ICPBasedRegistration()
: m_MaximumIterations(ICPBasedRegistrationConstants::DEFAULT_MAX_ITERATIONS)
, m_MaximumNumberOfLandmarkPointsToUse(ICPBasedRegistrationConstants::DEFAULT_MAX_POINTS)
, m_TLSIterations(ICPBasedRegistrationConstants::DEFAULT_TLS_ITERATIONS)
, m_TLSPercentage(ICPBasedRegistrationConstants::DEFAULT_TLS_PERCENTAGE)
{
}


//-----------------------------------------------------------------------------
ICPBasedRegistration::~ICPBasedRegistration()
{
}


//-----------------------------------------------------------------------------
double ICPBasedRegistration::Update(const mitk::DataNode::Pointer& fixedNode,
                                    const mitk::DataNode::Pointer& movingNode,
                                    vtkMatrix4x4& transformMovingToFixed,
                                    const mitk::DataNode::Pointer& cameraNode,
                                    bool flipNormals
                                   )
{
  vtkSmartPointer<vtkPolyData> fixedPoly = vtkSmartPointer<vtkPolyData>::New();
  niftk::NodeToPolyData(fixedNode, *fixedPoly);

  vtkSmartPointer<vtkPolyData> movingPoly = vtkSmartPointer<vtkPolyData>::New();
  niftk::NodeToPolyData(movingNode, *movingPoly, cameraNode, flipNormals);

  return RunVTKICP(fixedPoly, movingPoly, transformMovingToFixed);
}


//-----------------------------------------------------------------------------
double ICPBasedRegistration::Update(const std::vector<mitk::DataNode::Pointer>& fixedNodes,
                                    const std::vector<mitk::DataNode::Pointer>& movingNodes,
                                    vtkMatrix4x4& transformMovingToFixed,
                                    const mitk::DataNode::Pointer& cameraNode,
                                    bool flipNormals
                                   )
{

  if (fixedNodes.empty())
  {
    mitkThrow() << "Fixed node list is empty.";
  }

  if (movingNodes.empty())
  {
    mitkThrow() << "Moving node list is empty.";
  }

  vtkSmartPointer<vtkPolyData> mergedFixedPolyData
    = niftk::MergePolyData(fixedNodes);

  vtkSmartPointer<vtkPolyData> mergedMovingPolyData
    = niftk::MergePolyData(movingNodes, cameraNode, flipNormals);

  return RunVTKICP(mergedFixedPolyData, mergedMovingPolyData, transformMovingToFixed);
}


//-----------------------------------------------------------------------------
double ICPBasedRegistration::RunVTKICP(vtkPolyData* fixedPoly,
                                       vtkPolyData* movingPoly,
                                       vtkMatrix4x4& transformMovingToFixed)
{
  if (fixedPoly == nullptr)
  {
    mitkThrow() << "In ICPBasedRegistration::RunVTKICP, fixedPoly is NULL";
  }

  if (movingPoly == nullptr)
  {
    mitkThrow() << "In ICPBasedRegistration::RunVTKICP, movingPoly is NULL";
  }

  double residual = std::numeric_limits<double>::max();

  try
  {
    niftk::VTKIterativeClosestPoint *icp = new  niftk::VTKIterativeClosestPoint();
    icp->SetICPMaxLandmarks(m_MaximumNumberOfLandmarkPointsToUse);
    icp->SetICPMaxIterations(m_MaximumIterations);
    icp->SetTLSIterations(m_TLSIterations);
    icp->SetTLSPercentage(m_TLSPercentage);
    icp->SetSource(movingPoly);
    icp->SetTarget(fixedPoly);

    residual = icp->Run();

    // Retrieve transformation
    vtkSmartPointer<vtkMatrix4x4> temp = icp->GetTransform();
    transformMovingToFixed.DeepCopy(temp);

    // Tidy up
    delete icp;
  }
  catch (const std::exception& e)
  {
    mitkThrow() << e.what();
  }

  return residual;
}

} // end namespace
