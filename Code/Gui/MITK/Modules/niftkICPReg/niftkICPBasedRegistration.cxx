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
#include <niftkVTKBackfaceCullingFilter.h>
#include <niftkVTKFunctions.h>
#include <vtkCellArray.h>
#include <vtkPoints.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkPolyDataNormals.h>
#include <mitkFileIOUtils.h>
#include <mitkCoordinateAxesData.h>
#include <mitkAffineTransformDataNodeProperty.h>
#include <mitkDataStorageUtils.h>
#include <mitkSurface.h>

namespace niftk
{

//-----------------------------------------------------------------------------
ICPBasedRegistration::ICPBasedRegistration()
: m_MaximumIterations(ICPBasedRegistrationConstants::DEFAULT_MAX_ITERATIONS)
, m_MaximumNumberOfLandmarkPointsToUse(ICPBasedRegistrationConstants::DEFAULT_MAX_POINTS)
, m_CameraNode(NULL)
, m_FlipNormals(false)
{
}


//-----------------------------------------------------------------------------
ICPBasedRegistration::~ICPBasedRegistration()
{
}


//-----------------------------------------------------------------------------
void ICPBasedRegistration::RunVTKICP(vtkPolyData* fixedPoly,
                                     vtkPolyData* movingPoly,
                                     vtkMatrix4x4& transformMovingToFixed)
{
  if (fixedPoly == NULL)
  {
    mitkThrow() << "In ICPBasedRegistration::RunVTKICP, fixedPoly is NULL";
  }

  if (movingPoly == NULL)
  {
    mitkThrow() << "In ICPBasedRegistration::RunVTKICP, movingPoly is NULL";
  }

  niftk::VTKIterativeClosestPoint *icp = new  niftk::VTKIterativeClosestPoint();
  icp->SetICPMaxLandmarks(m_MaximumNumberOfLandmarkPointsToUse);
  icp->SetICPMaxIterations(m_MaximumIterations);
  icp->SetSource(movingPoly);
  icp->SetTarget(fixedPoly);

  // Throws exception if fails.
  icp->Run();

  // Retrieve transformation
  vtkSmartPointer<vtkMatrix4x4> temp = icp->GetTransform();
  transformMovingToFixed.DeepCopy(temp);

  // Tidy up
  delete icp;
}


//-----------------------------------------------------------------------------
void ICPBasedRegistration::Update(const mitk::DataNode::Pointer fixedNode,
                                  const mitk::DataNode::Pointer movingNode,
                                  vtkMatrix4x4& transformMovingToFixed)
{
  vtkSmartPointer<vtkPolyData> fixedPoly = vtkSmartPointer<vtkPolyData>::New();
  NodeToPolyData ( fixedNode, *fixedPoly);

  vtkSmartPointer<vtkPolyData> movingPoly = vtkSmartPointer<vtkPolyData>::New();
  NodeToPolyData ( movingNode, *movingPoly, m_CameraNode, m_FlipNormals);

  RunVTKICP ( fixedPoly, movingPoly, transformMovingToFixed );
}


//-----------------------------------------------------------------------------
void ICPBasedRegistration::PointSetToPolyData ( const mitk::PointSet::Pointer& pointsIn, vtkPolyData& polyOut)
{
  vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();

  for (mitk::PointSet::PointsConstIterator i = pointsIn->Begin(); i != pointsIn->End(); ++i)
  {
    // we need the point in world-coordinates! i.e. take its index-to-world transformation into account.
    // so instead of i->Value() we go via GetPointIfExists(i->Id(), ...)
    mitk::PointSet::PointType p = i->Value();
    pointsIn->GetPointIfExists(i->Index(), &p);
    points->InsertNextPoint(p[0], p[1], p[2]);
  }
  if (pointsIn->GetSize() != points->GetNumberOfPoints())
  {
    mitkThrow() << "Number of points in mitk::PointSet (" << pointsIn->GetSize()
                << ") != number of points in output (" <<  points->GetNumberOfPoints()
                << ")" << std::endl;
  }

  // What happens if polyOut contains vertices, triangles, poly-lines etc.
  // I think its safer to start again, so I'm calling Initialize().
  polyOut.Initialize();
  polyOut.SetPoints(points);
}


//-----------------------------------------------------------------------------
void ICPBasedRegistration::NodeToPolyData (
    const mitk::DataNode::Pointer& node , vtkPolyData& polyOut,
    const mitk::DataNode::Pointer& cameraNode, bool flipNormals)
{
  if (node.IsNull())
  {
    mitkThrow() << "In ICPBasedRegistration::NodeToPolyData, node is NULL";
  }

  mitk::PointSet::Pointer points = dynamic_cast<mitk::PointSet*>(node->GetData());
  mitk::Surface::Pointer surface = dynamic_cast<mitk::Surface*>(node->GetData());

  if (points.IsNotNull())
  {
    PointSetToPolyData (points, polyOut);
  }
  else if (surface.IsNotNull())
  {
    // No smartpointer for polyTemp!
    // mitk is handing us back a raw pointer, and uses raw pointers internally.
    vtkPolyData *polyTemp = surface->GetVtkPolyData();

    vtkSmartPointer<vtkMatrix4x4> indexToWorld = vtkSmartPointer<vtkMatrix4x4>::New();
    mitk::GetCurrentTransformFromNode(node, *indexToWorld);

    vtkSmartPointer<vtkTransform> transform = vtkSmartPointer<vtkTransform>::New();
    transform->SetMatrix(indexToWorld);

    if (cameraNode.IsNull())
    {
      vtkSmartPointer<vtkTransformPolyDataFilter> transformFilter= vtkSmartPointer<vtkTransformPolyDataFilter>::New();
      transformFilter->SetInputDataObject(polyTemp);
      transformFilter->SetOutput(&polyOut);
      transformFilter->SetTransform(transform);
      transformFilter->Update();
    }
    else
    {
      vtkSmartPointer<vtkPolyData> transformedPoly = vtkSmartPointer<vtkPolyData>::New();
      vtkSmartPointer<vtkTransformPolyDataFilter> transformFilter= vtkSmartPointer<vtkTransformPolyDataFilter>::New();
      transformFilter->SetInputDataObject(polyTemp);
      transformFilter->SetOutput(transformedPoly);
      transformFilter->SetTransform(transform);
      transformFilter->Update();

      vtkSmartPointer<vtkPolyData> normalsPoly = vtkSmartPointer<vtkPolyData>::New();
      vtkSmartPointer<vtkPolyDataNormals>   normalFilter = vtkSmartPointer<vtkPolyDataNormals>::New();
      normalFilter->SetInputDataObject(transformedPoly);
      normalFilter->SetOutput(normalsPoly);
      normalFilter->ComputeCellNormalsOn();
      normalFilter->ComputePointNormalsOn();
      normalFilter->AutoOrientNormalsOn();
      normalFilter->ConsistencyOn();
      normalFilter->SetFlipNormals(flipNormals ? 1 : 0);
      normalFilter->Update();

      vtkSmartPointer<vtkMatrix4x4> camtxf = vtkSmartPointer<vtkMatrix4x4>::New();
      mitk::GetCurrentTransformFromNode(cameraNode, *camtxf);

      vtkSmartPointer<niftk::BackfaceCullingFilter> backfacecullingfilter =
          vtkSmartPointer<niftk::BackfaceCullingFilter>::New();
      backfacecullingfilter->SetInputDataObject(normalsPoly);
      backfacecullingfilter->SetOutput(&polyOut);
      backfacecullingfilter->SetCameraPosition(camtxf);
      // this should call Update() instead of Execute().
      // but vtk6 has changed in some way that the filter's Execute() is no longer called.
      backfacecullingfilter->Execute();
    }
  }
  else
  {
    mitkThrow() << "In ICPBasedRegistration::NodeToPolyData, \
                   node is neither mitk::PointSet or mitk::Surface";
  }
}

//-----------------------------------------------------------------------------
} // end namespace
