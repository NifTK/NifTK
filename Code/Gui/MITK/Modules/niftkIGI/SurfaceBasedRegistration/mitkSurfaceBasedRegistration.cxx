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
,m_Matrix(NULL)
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
  m_Matrix = vtkMatrix4x4::New();
  m_Matrix->DeepCopy(temp);
}

void SurfaceBasedRegistration::Update(const mitk::DataNode* fixedNode, 
                                       const mitk::DataNode* movingNode,
                                       vtkMatrix4x4 * transformMovingToFixed)
{
  if ( m_Method == VTK_ICP )
  {
    vtkPolyData * fixedPoly = vtkPolyData::New();
    NodeToPolyData ( fixedNode, fixedPoly );
    vtkPolyData * movingPoly = vtkPolyData::New();
    NodeToPolyData ( movingNode, movingPoly );
    RunVTKICP ( fixedPoly, movingPoly, transformMovingToFixed );
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
  int numberOfPoints = PointsIn->GetSize();
  int i = 0 ;
  int PointsFound = 0 ;
  while ( PointsFound < numberOfPoints )
  {
    mitk::Point3D point;
    if ( PointsIn->GetPointIfExists(i,&point) )
    {
      points->InsertNextPoint(point[0],point[1],point[2]);
      PointsFound ++ ;
    }
    i++;
  }
  PolyOut->SetPoints(points);
}

void SurfaceBasedRegistration::NodeToPolyData (const  mitk::DataNode* node, vtkPolyData* PolyOut )
{
  mitk::Surface::Pointer Surface = NULL;
  mitk::PointSet::Pointer Points = NULL;
  Surface = dynamic_cast<mitk::Surface*>(node->GetData());
  Points = dynamic_cast<mitk::PointSet*>(node->GetData());
  if ( Surface.IsNull() ) 
  {
    PointSetToPolyData ( Points,PolyOut );
  }
  else
  {
    vtkPolyData * polytemp = vtkPolyData::New();
    polytemp=Surface->GetVtkPolyData();
    vtkMatrix4x4 * indexToWorld = vtkMatrix4x4::New();
    GetCurrentTransform(node,indexToWorld);
    vtkTransform * transform = vtkTransform::New();
    transform->SetMatrix(indexToWorld);
    vtkTransformPolyDataFilter * transformFilter= vtkTransformPolyDataFilter::New();
    transformFilter->SetInput(polytemp);
    transformFilter->SetOutput(PolyOut);
    transformFilter->SetTransform(transform);
    transformFilter->Update();

  }
}
  
void SurfaceBasedRegistration::ApplyTransform (mitk::DataNode::Pointer node)
{
  ApplyTransform(node, m_Matrix);
}
void SurfaceBasedRegistration::ApplyTransform (mitk::DataNode::Pointer node , vtkMatrix4x4 * matrix)
{
  vtkMatrix4x4 * CurrentMatrix = vtkMatrix4x4::New();
  GetCurrentTransform (node , CurrentMatrix );
  vtkMatrix4x4 * NewMatrix = vtkMatrix4x4::New();
  matrix->Multiply4x4(matrix, CurrentMatrix, NewMatrix);
  mitk::CoordinateAxesData* transform = dynamic_cast<mitk::CoordinateAxesData*>(node->GetData());

  if (transform != NULL)
  {
    mitk::AffineTransformDataNodeProperty::Pointer property = dynamic_cast<mitk::AffineTransformDataNodeProperty*>(node->GetProperty("niftk.transform"));
    if (property.IsNull())
    {
      MITK_ERROR << "LiverSurgeryManager::SetTransformation the node " << node->GetName() << " does not contain the niftk.transform property" << std::endl;
      return;
    }

    transform->SetVtkMatrix(*NewMatrix);
    transform->Modified();

    property->SetTransform(*NewMatrix);
    property->Modified();
  }
  else
  {
    mitk::Geometry3D::Pointer geometry = node->GetData()->GetGeometry();
    if (geometry.IsNotNull())
    {
      geometry->SetIndexToWorldTransformByVtkMatrix(NewMatrix);
      geometry->Modified();
    }
  }
}

void SurfaceBasedRegistration::GetCurrentTransform (const mitk::DataNode* node, vtkMatrix4x4* Matrix)
{
  mitk::AffineTransform3D::Pointer affineTransform = node->GetData()->GetGeometry()->GetIndexToWorldTransform();
  itk::Matrix<float, 3, 3>  matrix;
  itk::Vector<float, 3> offset;
  matrix = affineTransform->GetMatrix();
  offset = affineTransform->GetOffset();

  Matrix->Identity();
  for ( int i = 0 ; i < 3 ; i ++ ) 
  {
    for ( int j = 0 ; j < 3 ; j ++ )
    {
      Matrix->SetElement (i,j,matrix[i][j]);
    }
  }
  for ( int i = 0 ; i < 3 ; i ++ ) 
  {
    Matrix->SetElement (i, 3, offset[i]);
  }

}
} // end namespace

