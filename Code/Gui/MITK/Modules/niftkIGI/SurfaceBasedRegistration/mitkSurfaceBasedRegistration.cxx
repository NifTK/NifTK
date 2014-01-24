/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkSurfaceBasedRegistration.h"
#include <niftkVTKIterativeClosestPoint.h>
#include <vtkCellArray.h>
#include <vtkPoints.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>
#include <mitkFileIOUtils.h>
#include <mitkCoordinateAxesData.h>
#include <mitkAffineTransformDataNodeProperty.h>
#include <niftkVTKFunctions.h>
#include <niftkVTKBackfaceCullingFilter.h>
#include <mitkDataStorageUtils.h>
#include <vtkPolyDataNormals.h>


namespace mitk
{
 
const int SurfaceBasedRegistration::DEFAULT_MAX_ITERATIONS = 2000;
const int SurfaceBasedRegistration::DEFAULT_MAX_POINTS = 8000;
const bool SurfaceBasedRegistration::DEFAULT_USE_DEFORMABLE = false;

//-----------------------------------------------------------------------------
SurfaceBasedRegistration::SurfaceBasedRegistration()
: m_MaximumIterations(SurfaceBasedRegistration::DEFAULT_MAX_ITERATIONS)
, m_MaximumNumberOfLandmarkPointsToUse(SurfaceBasedRegistration::DEFAULT_MAX_POINTS)
, m_Method(VTK_ICP)
, m_FlipNormals(false)
, m_Matrix(NULL)
{
  m_Matrix = vtkMatrix4x4::New();
}


//-----------------------------------------------------------------------------
SurfaceBasedRegistration::~SurfaceBasedRegistration()
{
}


//-----------------------------------------------------------------------------
void SurfaceBasedRegistration::RunVTKICP(vtkPolyData* fixedPoly,
                                      vtkPolyData* movingPoly,
                                      vtkMatrix4x4& transformMovingToFixed)
{

  if (fixedPoly == NULL)
  {
    mitkThrow() << "In SurfaceBasedRegistration::RunVTKICP, fixedPoly is NULL";
  }

  if (movingPoly == NULL)
  {
    mitkThrow() << "In SurfaceBasedRegistration::RunVTKICP, movingPoly is NULL";
  }

  niftk::VTKIterativeClosestPoint *icp = new  niftk::VTKIterativeClosestPoint();
  icp->SetMaxLandmarks(m_MaximumNumberOfLandmarkPointsToUse);
  icp->SetMaxIterations(m_MaximumIterations);
  icp->SetSource(movingPoly);
  icp->SetTarget(fixedPoly);
  icp->Run();

  vtkMatrix4x4 *temp = icp->GetTransform();

  transformMovingToFixed.DeepCopy(temp);
  m_Matrix->DeepCopy(temp);

  delete icp;
}


//-----------------------------------------------------------------------------
void SurfaceBasedRegistration::Update(const mitk::DataNode::Pointer fixedNode,
                                       const mitk::DataNode::Pointer movingNode,
                                       vtkMatrix4x4& transformMovingToFixed)
{
  if ( m_Method == VTK_ICP )
  {
    vtkSmartPointer<vtkPolyData> fixedPoly = vtkPolyData::New();
    NodeToPolyData ( fixedNode, *fixedPoly);

    vtkSmartPointer<vtkPolyData> movingPoly = vtkPolyData::New();
    m_CulledPolyData = NodeToPolyData ( movingNode, *movingPoly, m_CameraNode, m_FlipNormals);

    //m_CulledPolyData = movingPoly;
    //RunVTKICP ( fixedPoly, movingPoly, transformMovingToFixed );
  }
  if ( m_Method == DEFORM )
  {
    // Not Implenented yet.
  }

}


//-----------------------------------------------------------------------------
void SurfaceBasedRegistration::PointSetToPolyData ( const mitk::PointSet::Pointer& pointsIn, vtkPolyData& polyOut)
{
  vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();

  for (mitk::PointSet::PointsConstIterator i = pointsIn->Begin(); i != pointsIn->End(); ++i)
  {
    const mitk::PointSet::PointType& p = i->Value();
    points->InsertNextPoint(p[0], p[1], p[2]);
  }
  // sanity check
  assert(pointsIn->GetSize() == points->GetNumberOfPoints());

  polyOut.SetPoints(points);
}


//-----------------------------------------------------------------------------
vtkSmartPointer<vtkPolyData> SurfaceBasedRegistration::NodeToPolyData ( const mitk::DataNode::Pointer& node , vtkPolyData& polyOut, const mitk::DataNode::Pointer& cameranode, bool flipnormals)
{
  vtkSmartPointer<vtkPolyData>    result;

  if (node.IsNull())
  {
    mitkThrow() << "In SurfaceBasedRegistration::NodeToPolyData, node is NULL";
  }

  mitk::PointSet::Pointer points = dynamic_cast<mitk::PointSet*>(node->GetData());
  mitk::Surface::Pointer surface = dynamic_cast<mitk::Surface*>(node->GetData());

  if (points.IsNotNull())
  {
    PointSetToPolyData ( points, polyOut );
  }
  else if (surface.IsNotNull())
  {
    // no smartpointer for polytemp!
    // mitk is handing us back a raw pointer, and uses raw pointers internally.
    vtkPolyData *polytemp = surface->GetVtkPolyData();

    vtkSmartPointer<vtkMatrix4x4> indexToWorld = vtkMatrix4x4::New();
    mitk::GetCurrentTransformFromNode(node, *indexToWorld);

    vtkSmartPointer<vtkTransform> transform = vtkTransform::New();
    transform->SetMatrix(indexToWorld);

    if (cameranode.IsNull())
    {
      vtkSmartPointer<vtkTransformPolyDataFilter> transformFilter= vtkTransformPolyDataFilter::New();
      transformFilter->SetInput(polytemp);
      transformFilter->SetOutput(&polyOut);
      transformFilter->SetTransform(transform);
      transformFilter->Update();
    }
    else
    {
      vtkSmartPointer<vtkPolyData> transformedpoly = vtkPolyData::New();
      vtkSmartPointer<vtkTransformPolyDataFilter> transformFilter= vtkTransformPolyDataFilter::New();
      transformFilter->SetInput(polytemp);
      transformFilter->SetOutput(transformedpoly);
      transformFilter->SetTransform(transform);
      transformFilter->Update();

      vtkSmartPointer<vtkPolyData> normalspoly = vtkPolyData::New();
      vtkSmartPointer<vtkPolyDataNormals>   normalfilter = vtkPolyDataNormals::New();
      normalfilter->SetInput(transformedpoly);
      normalfilter->SetOutput(normalspoly);
      normalfilter->ComputeCellNormalsOn();
      normalfilter->ComputePointNormalsOn();
      normalfilter->AutoOrientNormalsOn();
      normalfilter->ConsistencyOn();
      normalfilter->SetFlipNormals(flipnormals ? 1 : 0);
      normalfilter->Update();

      vtkSmartPointer<vtkMatrix4x4> camtxf = vtkMatrix4x4::New();
      mitk::GetCurrentTransformFromNode(cameranode, *camtxf);

      vtkSmartPointer<niftk::BackfaceCullingFilter>   backfacecullingfilter = niftk::BackfaceCullingFilter::New();
      backfacecullingfilter->SetInput(normalspoly);
      backfacecullingfilter->SetOutput(&polyOut);
      backfacecullingfilter->SetCameraPosition(camtxf);
      backfacecullingfilter->Update();

      result = &polyOut;
    }
  }
  else
  {
    mitkThrow() << "In SurfaceBasedRegistration::NodeToPolyData, node is neither mitk::PointSet or mitk::Surface";
  }

  return result;
}

//-----------------------------------------------------------------------------
} // end namespace

