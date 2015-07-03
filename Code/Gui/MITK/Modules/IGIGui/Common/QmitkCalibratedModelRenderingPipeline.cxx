/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkCalibratedModelRenderingPipeline.h"
#include <QApplication>
#include <QMessageBox>
#include <QThread>
#include <vtkPoints.h>
#include <vtkPointData.h>
#include <vtkDoubleArray.h>
#include <mitkPointSet.h>
#include <mitkIOUtil.h>


//-----------------------------------------------------------------------------
QmitkCalibratedModelRenderingPipeline::QmitkCalibratedModelRenderingPipeline(const std::string& name,
  const mitk::Point2I& windowSize,
  const mitk::Point2I& calibratedWindowSize,
  const std::string& leftIntrinsicsFileName,
  const std::string& rightIntrinsicsFileName,
  const std::string& visualisationModelFileName,
  const std::string& rightToLeftFileName,
  const std::string& textureFileName,
  const std::string& trackingModelFileName,
  const float& trackingGlyphRadius,
  const std::string &outputData,
  QWidget *parent)
: QVTKWidget(parent)
, m_Pipeline(name, windowSize, calibratedWindowSize, leftIntrinsicsFileName, rightIntrinsicsFileName, visualisationModelFileName, rightToLeftFileName, textureFileName, trackingModelFileName, trackingGlyphRadius)
, m_OutputData(outputData)
{
  m_WindowSize[0] = windowSize[0];
  m_WindowSize[1] = windowSize[1];
  m_CalibratedWindowSize[0] = calibratedWindowSize[0];
  m_CalibratedWindowSize[1] = calibratedWindowSize[1];
  this->SetRenderWindow(m_Pipeline.GetRenderWindow());
  this->GetInteractor()->Disable();
}


//-----------------------------------------------------------------------------
QmitkCalibratedModelRenderingPipeline::~QmitkCalibratedModelRenderingPipeline()
{
}


//-----------------------------------------------------------------------------
void QmitkCalibratedModelRenderingPipeline::SetModelToWorldMatrix(const vtkMatrix4x4& modelToWorld)
{
  m_Pipeline.SetModelToWorldMatrix(modelToWorld);
}


//-----------------------------------------------------------------------------
void QmitkCalibratedModelRenderingPipeline::SetModelToWorldTransform(const std::vector<float>& transform)
{
  m_Pipeline.SetModelToWorldTransform(transform);
}


//-----------------------------------------------------------------------------
void QmitkCalibratedModelRenderingPipeline::SetCameraToWorldMatrix(const vtkMatrix4x4& cameraToWorld)
{
  m_Pipeline.SetCameraToWorldMatrix(cameraToWorld);
}


//-----------------------------------------------------------------------------
void QmitkCalibratedModelRenderingPipeline::SetCameraToWorldTransform(const std::vector<float>& transform)
{
  m_Pipeline.SetCameraToWorldTransform(transform);
}


//-----------------------------------------------------------------------------
void QmitkCalibratedModelRenderingPipeline::SetWorldToCameraMatrix(const vtkMatrix4x4& worldToCamera)
{
  m_Pipeline.SetWorldToCameraMatrix(worldToCamera);
}


//-----------------------------------------------------------------------------
void QmitkCalibratedModelRenderingPipeline::SetWorldToCameraTransform(const std::vector<float>& transform)
{
  m_Pipeline.SetWorldToCameraTransform(transform);
}


//-----------------------------------------------------------------------------
void QmitkCalibratedModelRenderingPipeline::SetIsRightHandCamera(const bool& isRight)
{
  m_Pipeline.SetIsRightHandCamera(isRight);
}


//-----------------------------------------------------------------------------
void QmitkCalibratedModelRenderingPipeline::Render()
{
  m_Pipeline.Render();
}


//-----------------------------------------------------------------------------
void QmitkCalibratedModelRenderingPipeline::SaveLeftImage()
{
  m_Pipeline.SetIsRightHandCamera(false);
  m_Pipeline.Render();
  m_Pipeline.DumpScreen(m_OutputData + std::string(".left.png"));
}


//-----------------------------------------------------------------------------
void QmitkCalibratedModelRenderingPipeline::SaveRightImage()
{
  m_Pipeline.SetIsRightHandCamera(true);
  m_Pipeline.Render();
  m_Pipeline.DumpScreen(m_OutputData + std::string(".right.png"));
}


//-----------------------------------------------------------------------------
void QmitkCalibratedModelRenderingPipeline::SaveData()
{
  this->SaveLeftImage();
  this->SaveRightImage();
  m_Pipeline.SaveModelToWorld(m_OutputData + std::string(".Model2World.4x4"));
  m_Pipeline.SaveCameraToWorld(m_OutputData + std::string(".Camera2World.4x4"));

  // Now, iterate over all points in tracking model, and project to 2D.
  vtkPolyData* trackingModel = m_Pipeline.GetTrackingModel();
  vtkPoints *points = trackingModel->GetPoints();
  vtkDoubleArray *normals = static_cast<vtkDoubleArray*>(trackingModel->GetPointData()->GetNormals());
  vtkIntArray *ids = static_cast<vtkIntArray*>(trackingModel->GetPointData()->GetScalars());

  assert(points);
  assert(normals);
  assert(ids);
  assert(points->GetNumberOfPoints() == normals->GetNumberOfTuples());
  assert(points->GetNumberOfPoints() == ids->GetNumberOfTuples());

  int id;
  double point[3];
  double cameraPoint[3];
  double normal[3];
  double imageLeft[2];
  double imageRight[2];

  std::ofstream outputFile;
  outputFile.open(m_OutputData.c_str(), std::ios::out);

  mitk::PointSet::Pointer pointSet = mitk::PointSet::New();
  mitk::Point3D mitkPoint;

  mitk::Point2D scaleFactors;
  scaleFactors[0] = static_cast<double>(m_WindowSize[0])/static_cast<double>(m_CalibratedWindowSize[0]);
  scaleFactors[1] = static_cast<double>(m_WindowSize[1])/static_cast<double>(m_CalibratedWindowSize[1]);

  for (vtkIdType i = 0; i < points->GetNumberOfPoints(); i++)
  {
    points->GetPoint(i, point);
    normals->GetTuple(i, normal);
    id = ids->GetValue(i);

    m_Pipeline.SetIsRightHandCamera(false);
    m_Pipeline.ProjectPoint(point, imageLeft);
    bool isFacingLeft = m_Pipeline.IsFacingCamera(normal);

    m_Pipeline.ProjectToCameraSpace(point, cameraPoint);

    m_Pipeline.SetIsRightHandCamera(true);
    m_Pipeline.ProjectPoint(point, imageRight);
    bool isFacingRight = m_Pipeline.IsFacingCamera(normal);

    if (isFacingLeft && isFacingRight)
    {
      outputFile << id << " " << point[0] << " " << point[1] << " " << point[2] << " " << cameraPoint[0] << " " << cameraPoint[1] << " " << cameraPoint[2] << " " << imageLeft[0]*scaleFactors[0] << " " << imageLeft[1]*scaleFactors[1] << " " << imageRight[0]*scaleFactors[0] << " " << imageRight[1]*scaleFactors[1] << std::endl;
      mitkPoint[0] = cameraPoint[0];
      mitkPoint[1] = cameraPoint[1];
      mitkPoint[2] = cameraPoint[2];
      pointSet->InsertPoint(id, mitkPoint);
    }
  }

  outputFile.close();

  mitk::IOUtil::Save(pointSet, m_OutputData + std::string(".mps"));
}

