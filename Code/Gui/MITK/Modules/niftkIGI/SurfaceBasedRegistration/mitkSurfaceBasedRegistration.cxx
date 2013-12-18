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
#include <mitkDataStorageUtils.h>

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
    NodeToPolyData ( movingNode, *movingPoly);

    RunVTKICP ( fixedPoly, movingPoly, transformMovingToFixed );
  }
  if ( m_Method == DEFORM )
  {
    // Not Implenented yet.
  }

}


//-----------------------------------------------------------------------------
void SurfaceBasedRegistration::PointSetToPolyData ( const mitk::PointSet::Pointer pointsIn, vtkPolyData& polyOut)
{
  vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
  int numberOfPoints = pointsIn->GetSize();

  int i = 0 ;
  int pointsFound = 0 ;
  while ( pointsFound < numberOfPoints )
  {
    mitk::Point3D point;
    if ( pointsIn->GetPointIfExists(i, &point))
    {
      points->InsertNextPoint(point[0], point[1], point[2]);
      pointsFound++;
    }
    i++;
  }
  polyOut.SetPoints(points);
}


//-----------------------------------------------------------------------------
void SurfaceBasedRegistration::NodeToPolyData ( const mitk::DataNode::Pointer node , vtkPolyData& polyOut)
{
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
    vtkPolyData *polytemp = surface->GetVtkPolyData();

    vtkSmartPointer<vtkMatrix4x4> indexToWorld = vtkMatrix4x4::New();
    mitk::GetCurrentTransformFromNode(node, *indexToWorld);

    vtkSmartPointer<vtkTransform> transform = vtkTransform::New();
    transform->SetMatrix(indexToWorld);

    vtkTransformPolyDataFilter * transformFilter= vtkTransformPolyDataFilter::New();
    transformFilter->SetInput(polytemp);
    transformFilter->SetOutput(&polyOut);
    transformFilter->SetTransform(transform);
    transformFilter->Update();
  }
  else
  {
    mitkThrow() << "In SurfaceBasedRegistration::NodeToPolyData, node is neither mitk::PointSet or mitk::Surface";
  }
}

//-----------------------------------------------------------------------------
} // end namespace

