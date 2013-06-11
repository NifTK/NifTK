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
#include <mitkFileIOUtils.h>

namespace mitk
{
 
const int SurfaceBasedRegistration::DEFAULT_MAX_ITERATIONS = 100;
const int SurfaceBasedRegistration::DEFAULT_MAX_POINTS = 100;
const bool SurfaceBasedRegistration::DEFAULT_USE_DEFORMABLE = false;
//-----------------------------------------------------------------------------
SurfaceBasedRegistration::SurfaceBasedRegistration()
:m_MaximumIterations(50)
,m_MaximumNumberOfLandmarkPointsToUse(200)
,m_Method(VTK_ICP)
{
}


//-----------------------------------------------------------------------------
SurfaceBasedRegistration::~SurfaceBasedRegistration()
{
}


//-----------------------------------------------------------------------------
void SurfaceBasedRegistration::Update(const mitk::Surface::Pointer fixedNode,
                                      const mitk::Surface::Pointer movingNode,
                                      vtkMatrix4x4* transformMovingToFixed)
{
  if ( m_Method == VTK_ICP ) 
  {
    RunVTKICP(fixedNode->GetVtkPolyData() , movingNode->GetVtkPolyData(), transformMovingToFixed);
  }
  if ( m_Method == DEFORM ) 
  {
    //Not Implemented
  }
}

//-----------------------------------------------------------------------------
void SurfaceBasedRegistration::RunVTKICP(vtkPolyData* fixedPoly,
                                      vtkPolyData* movingPoly,
                                      vtkMatrix4x4 * transformMovingToFixed)
{
  niftkVTKIterativeClosestPoint * icp = new  niftkVTKIterativeClosestPoint();
  icp->SetMaxLandmarks(m_MaximumNumberOfLandmarkPointsToUse);
  icp->SetMaxIterations(m_MaximumIterations);
  icp->SetSource(movingPoly);
  icp->SetTarget(fixedPoly);

  icp->Run();
  vtkMatrix4x4 * temp;
  temp = icp->GetTransform();
  transformMovingToFixed->DeepCopy(temp);
}


//-----------------------------------------------------------------------------
void SurfaceBasedRegistration::Update(const mitk::PointSet::Pointer fixedNode,
                                      const mitk::Surface::Pointer movingNode,
                                      vtkMatrix4x4 * transformMovingToFixed)
{
  if ( m_Method == VTK_ICP )
  {
    
    vtkPolyData * fixedPoly = vtkPolyData::New();
    PointSetToPolyData ( fixedNode, fixedPoly );
    RunVTKICP ( fixedPoly, movingNode->GetVtkPolyData(), transformMovingToFixed );
  }
  if ( m_Method == DEFORM )
  {
    //Not Implenented
  }
}

//-----------------------------------------------------------------------------
void SurfaceBasedRegistration::PointSetToPolyData (const  mitk::PointSet::Pointer PointsIn, vtkPolyData* PolyOut )
{
  vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
  vtkSmartPointer<vtkCellArray> verts = vtkSmartPointer<vtkCellArray>::New();
  int numberOfPoints = PointsIn->GetSize();
  for ( int i = 0 ; i < numberOfPoints ; i++ )
  {
    mitk::Point3D point = PointsIn->GetPoint(i);
    vtkIdType id = points->InsertNextPoint(point[0],point[1],point[2]);
    verts->InsertNextCell(1,&id);
  }
  PolyOut->SetPoints(points);
  PolyOut->SetVerts(verts);
}

} // end namespace

