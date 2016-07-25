/*===================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center,
Division of Medical and Biological Informatics.
All rights reserved.

This software is distributed WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR
A PARTICULAR PURPOSE.

See LICENSE.txt or http://www.mitk.org for details.

===================================================================*/


#include "niftkAffineTransformDataInteractor3D.h"

#include <mitkRotationOperation.h>
#include <mitkBoundingObject.h>
#include <mitkInteractionEvent.h>
#include <mitkInteractionConst.h>
#include <mitkInteractionPositionEvent.h>

#include <vtkInteractorObserver.h>
#include <vtkCamera.h>
#include <vtkPointData.h>

#include <QDebug>


namespace niftk
{

//how precise must the user pick the point
//default value
AffineTransformDataInteractor3D::AffineTransformDataInteractor3D()
: m_InteractionMode(INTERACTION_MODE_TRANSLATION)
, m_AxesFixed(false)
, m_InitialPickedDisplayPoint(mitk::Point2D())
, m_CurrentlyPickedDisplayPoint(mitk::Point2D())
, m_OriginalGeometry(mitk::Geometry3D::New())
, m_UpdatedGeometry(mitk::Geometry3D::New())
, m_CurrentRenderer(NULL)
, m_CurrentVtkRenderer(NULL)
, m_CurrentCamera(NULL)
, m_BoundingObjectNode(NULL)
{
  // Initialize vector arithmetic
  m_ObjectNormal[0] = 0.0;
  m_ObjectNormal[1] = 0.0;
  m_ObjectNormal[2] = 1.0;
}


AffineTransformDataInteractor3D::~AffineTransformDataInteractor3D()
{
}


void AffineTransformDataInteractor3D::ConnectActionsAndFunctions()
{
  CONNECT_CONDITION("overObject", CheckObject);
  CONNECT_FUNCTION("selectObject", SelectObject);
  CONNECT_FUNCTION("deselectObject", DeselectObject);
  CONNECT_FUNCTION("initMove", InitMove);
  CONNECT_FUNCTION("move", Move);
  CONNECT_FUNCTION("accept", AcceptMove);
}


void AffineTransformDataInteractor3D::SetInteractionMode(unsigned int interactionMode)
{
  m_InteractionMode = interactionMode;
}


void AffineTransformDataInteractor3D::SetInteractionModeToTranslation()
{
  m_InteractionMode = INTERACTION_MODE_TRANSLATION;
}


void AffineTransformDataInteractor3D::SetInteractionModeToRotation()
{
  m_InteractionMode = INTERACTION_MODE_ROTATION;
}


unsigned int AffineTransformDataInteractor3D::GetInteractionMode() const
{
  return m_InteractionMode;
}


bool AffineTransformDataInteractor3D::UpdateCurrentRendererPointers(const mitk::InteractionEvent * interactionEvent)
{
  // Get Event and extract renderer
  if (interactionEvent == NULL)
  {
    return false;
  }

  m_CurrentRenderer = interactionEvent->GetSender();
  if (m_CurrentRenderer != NULL )
  {
    m_CurrentVtkRenderer = m_CurrentRenderer->GetVtkRenderer();

    if (m_CurrentVtkRenderer != NULL)
    {
      m_CurrentCamera = m_CurrentVtkRenderer->GetActiveCamera();
    }
    else
    {
      return false;
    }
  }
  else
  {
    return false;
  }

  return true;
}


bool AffineTransformDataInteractor3D::CheckObject(const mitk::InteractionEvent *interactionEvent)
{

  if (!UpdateCurrentRendererPointers(interactionEvent) ||
    this->GetDataNode()->GetData() == NULL || this->m_BoundingObjectNode == NULL)
  {
    return false;
  }

  // Check if we have a DisplayPositionEvent
  const mitk::InteractionPositionEvent* pe = dynamic_cast<const mitk::InteractionPositionEvent*>(interactionEvent);
  if (pe == NULL)
  {
    MITK_INFO << "no display position";
    //Could not resolve current display position: go back to start state
    return false;
  }

  mitk::Point3D currentlyPickedWorldPoint = pe->GetPositionInWorld();
  m_CurrentlyPickedDisplayPoint = pe->GetPointerPositionOnScreen();

  // Get the timestep to also support 3D+t
  int timeStep = 0;
  mitk::ScalarType timeInMS = 0.0;

  if (m_CurrentRenderer != NULL)
  {
    timeStep = m_CurrentRenderer->GetTimeStep(m_BoundingObjectNode->GetData());
    timeInMS = m_CurrentRenderer->GetTime();
  }

  mitk::BaseGeometry* geometry = m_BoundingObjectNode->GetData()->GetUpdatedTimeGeometry()->GetGeometryForTimeStep(timeStep);

  if (geometry->IsInside(currentlyPickedWorldPoint))
  {
    return true;
  }
  else
  {
    return false;
  }

  return true;
}


bool AffineTransformDataInteractor3D::SelectObject(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent)
{
  // Color object red
  this->GetDataNode()->SetColor(1.0, 0.0, 0.0);

  mitk::RenderingManager::Pointer renderManager = mitk::RenderingManager::GetInstance();
  renderManager->RequestUpdateAll();

  return true;
}


bool AffineTransformDataInteractor3D::DeselectObject(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent)
{
  // Color object white
  this->GetDataNode()->SetColor(1.0, 1.0, 1.0);

  mitk::RenderingManager::Pointer renderManager = mitk::RenderingManager::GetInstance();
  renderManager->RequestUpdateAll();

  return true;
}


bool AffineTransformDataInteractor3D::InitMove(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent)
{
  if (!UpdateCurrentRendererPointers(interactionEvent) || this->GetDataNode()->GetData() == NULL)
  {
    return false;
  }

  // Check if we have a InteractionPositionEvent
  const mitk::InteractionPositionEvent* pe = dynamic_cast<const mitk::InteractionPositionEvent*>(interactionEvent);
  if (pe == NULL)
  {
    return false;
  }

  m_InitialPickedDisplayPoint = m_CurrentlyPickedDisplayPoint;

  if (m_CurrentVtkRenderer != NULL)
  {
    vtkInteractorObserver::ComputeDisplayToWorld(
      m_CurrentVtkRenderer,
      m_InitialPickedDisplayPoint[0],
      m_InitialPickedDisplayPoint[1],
      0.0,
      m_InitialPickedPointWorld );
  }

  // Get the timestep to also support 3D+t
  int timeStep = 0;

  if (m_CurrentRenderer != NULL)
  {
    timeStep = m_CurrentRenderer->GetTimeStep(this->GetDataNode()->GetData());
  }

  // Make deep copy of current Geometry3D of the plane
  this->GetDataNode()->GetData()->UpdateOutputInformation();
  m_OriginalGeometry = dynamic_cast<mitk::BaseGeometry*>(this->GetDataNode()->GetData()->GetGeometry(timeStep)->Clone().GetPointer());

  return true;
}


bool AffineTransformDataInteractor3D::Move(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent)
{
  if (!UpdateCurrentRendererPointers(interactionEvent) || this->GetDataNode()->GetData() == NULL)
  {
    return false;
  }

  // Check if we have a InteractionPositionEvent
  const mitk::InteractionPositionEvent* pe = dynamic_cast<const mitk::InteractionPositionEvent*>(interactionEvent);
  if (pe == NULL)
  {
    return false;
  }

  m_CurrentlyPickedDisplayPoint = pe->GetPointerPositionOnScreen();

  mitk::Vector3D interactionMove;

  if (m_CurrentVtkRenderer != NULL)
  {
    vtkInteractorObserver::ComputeDisplayToWorld(
      m_CurrentVtkRenderer,
      m_CurrentlyPickedDisplayPoint[0],
      m_CurrentlyPickedDisplayPoint[1],
      0.0, //m_InitialInteractionPickedPoint[2],
      m_CurrentlyPickedPointWorld);
  }

  interactionMove[0] = m_CurrentlyPickedPointWorld[0] - m_InitialPickedPointWorld[0];
  interactionMove[1] = m_CurrentlyPickedPointWorld[1] - m_InitialPickedPointWorld[1];
  interactionMove[2] = m_CurrentlyPickedPointWorld[2] - m_InitialPickedPointWorld[2];

  // Get the timestep to also support 3D+t
  int timeStep = 0;

  if (m_CurrentRenderer != NULL)
    timeStep = m_CurrentRenderer->GetTimeStep(this->GetDataNode()->GetData());

  if (m_InteractionMode == INTERACTION_MODE_TRANSLATION)
  {
    mitk::Point3D origin = m_OriginalGeometry->GetOrigin();

    mitk::Vector3D transformedObjectNormal;
    this->GetDataNode()->GetData()->GetGeometry( timeStep )->IndexToWorld(m_ObjectNormal, transformedObjectNormal);

    if (m_AxesFixed == true)
    {
      this->GetDataNode()->GetData()->GetGeometry(timeStep)->SetOrigin(
        origin + transformedObjectNormal * (interactionMove * transformedObjectNormal));
    }
    else
    {
      this->GetDataNode()->GetData()->GetGeometry(timeStep)->SetOrigin(origin + interactionMove);
    }
  }
  else if (m_InteractionMode == INTERACTION_MODE_ROTATION)
  {
    if (m_CurrentCamera != NULL)
    {
      mitk::Vector3D rotationAxis;
      rotationAxis[0] = 0;
      rotationAxis[1] = 0;
      rotationAxis[2] = 0;

      if (m_AxesFixed == true)
      {
        rotationAxis[0] = m_ObjectNormal[0];
        rotationAxis[1] = m_ObjectNormal[1];
        rotationAxis[2] = m_ObjectNormal[2];
      }
      else
      {
        double vpn[3];
        m_CurrentCamera->GetViewPlaneNormal(vpn);

        mitk::Vector3D viewPlaneNormal;
        viewPlaneNormal[0] = vpn[0];
        viewPlaneNormal[1] = vpn[1];
        viewPlaneNormal[2] = vpn[2];

        //qDebug() <<"PlaneNormal: " <<vpn[0] <<vpn[1] <<vpn[2];

        rotationAxis = itk::CrossProduct(viewPlaneNormal, interactionMove);
        rotationAxis.Normalize();
      }

      //qDebug() <<"RotAxis: " <<rotationAxis[0] <<rotationAxis[1] <<rotationAxis[2];

      int *size = m_CurrentVtkRenderer->GetSize();
      double l2 =
        (m_CurrentlyPickedDisplayPoint[0] - m_InitialPickedDisplayPoint[0]) *
        (m_CurrentlyPickedDisplayPoint[0] - m_InitialPickedDisplayPoint[0]) +
        (m_CurrentlyPickedDisplayPoint[1] - m_InitialPickedDisplayPoint[1]) *
        (m_CurrentlyPickedDisplayPoint[1] - m_InitialPickedDisplayPoint[1]);

      double rotationAngle = 360.0 * sqrt(l2 / (size[0] * size[0] + size[1] * size[1]));

      // Use center of data bounding box as center of rotation
      mitk::Point3D rotationCenter;
      rotationCenter = m_OriginalGeometry->GetCenter();

      // Reset current Geometry3D to original state (pre-interaction) and
      // apply rotation
      mitk::RotationOperation op(mitk::OpROTATE, rotationCenter, rotationAxis, rotationAngle);
      mitk::BaseGeometry::Pointer newGeometry
        = dynamic_cast<mitk::BaseGeometry*>(this->GetDataNode()->GetData()->GetGeometry(timeStep)->Clone().GetPointer());

      if (newGeometry.IsNotNull())
      {
        newGeometry->mitk::BaseGeometry::ExecuteOperation(&op);
        mitk::TimeGeometry::Pointer timeGeometry = this->GetDataNode()->GetData()->GetTimeGeometry();
        bool succ = false;
        if (timeGeometry.IsNotNull() && timeGeometry->IsValidTimeStep(timeStep))
        {
          timeGeometry->SetTimeStepGeometry(newGeometry, timeStep);
          this->GetDataNode()->GetData()->Modified();
          this->GetDataNode()->Modified();
          this->GetDataNode()->Update();
        }
      }
    }
  }

  if (m_BoundingObjectNode != NULL)
  {
   static_cast<mitk::BoundingObject *>(m_BoundingObjectNode->GetData())->FitGeometry(this->GetDataNode()->GetData()->GetGeometry());
  }

  interactionEvent->GetSender()->GetRenderingManager()->RequestUpdateAll();

  return true;
}


vtkMatrix4x4* AffineTransformDataInteractor3D::GetUpdatedGeometry()
{
 return m_UpdatedGeometry->GetVtkMatrix();
}


bool AffineTransformDataInteractor3D::AcceptMove(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent)
{

  // Get the timestep to also support 3D+t
  int timeStep = 0;

  if (m_CurrentRenderer != NULL)
    timeStep = m_CurrentRenderer->GetTimeStep(this->GetDataNode()->GetData());

  // rest and compose transform in the correct order
  m_UpdatedGeometry->Initialize();

  vtkSmartPointer<vtkMatrix4x4> originalTransformation = m_OriginalGeometry->GetVtkMatrix();
  vtkSmartPointer<vtkMatrix4x4> invertedOriginalTransformation = vtkSmartPointer<vtkMatrix4x4>::New();
  vtkMatrix4x4::Invert(originalTransformation, invertedOriginalTransformation);


  m_UpdatedGeometry->Compose(invertedOriginalTransformation);

  mitk::BaseGeometry::Pointer nodeGeometry =
    dynamic_cast<mitk::BaseGeometry*>(this->GetDataNode()->GetData()->GetGeometry(timeStep)->Clone().GetPointer());
  m_UpdatedGeometry->Compose(nodeGeometry->GetVtkMatrix());

  emit transformReady();
  return true;
}


void AffineTransformDataInteractor3D::SetAxesFixed(bool on, int which)
{
  if (on == true)
  {
    m_AxesFixed = true;

    if (which == 0)
    {
      // Initialize vector arithmetic
      m_ObjectNormal[0] = 0.0;
      m_ObjectNormal[1] = 0.0;
      m_ObjectNormal[2] = 1.0;
    }
    else if (which == 1)
    {
      // Initialize vector arithmetic
      m_ObjectNormal[0] = 0.0;
      m_ObjectNormal[1] = 1.0;
      m_ObjectNormal[2] = 0.0;
    }
    else if (which == 2)
    {
      // Initialize vector arithmetic
      m_ObjectNormal[0] = 1.0;
      m_ObjectNormal[1] = 0.0;
      m_ObjectNormal[2] = 0.0;
    }
  }
  else
  {
    m_AxesFixed = false;
  }
}

}
