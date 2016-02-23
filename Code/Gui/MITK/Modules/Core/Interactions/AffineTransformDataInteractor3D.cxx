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


#include "AffineTransformDataInteractor3D.h"

#include <mitkRotationOperation.h>
#include <mitkBoundingObject.h>
#include <mitkInteractionEvent.h>
#include <mitkInteractionConst.h>

#include <vtkInteractorObserver.h>
#include <vtkCamera.h>
//#include <vtkPoints.h>
#include <vtkPointData.h>
#include <vtkTextProperty.h>
//#include <vtkDataArray.h>

#include <QDebug>


void mitk::AffineTransformDataInteractor3D::ConnectActionsAndFunctions()
{
  CONNECT_CONDITION("checkObject", CheckObject);
  CONNECT_FUNCTION("selectObject", SelectObject);
  CONNECT_FUNCTION("deselectObject", DeselectObject);
  CONNECT_FUNCTION("initMove", InitMove);
  CONNECT_FUNCTION("acceptMove", Move);
  CONNECT_FUNCTION("acceptMove", AcceptMove);
}

//how precise must the user pick the point
//default value
mitk::AffineTransformDataInteractor3D
::AffineTransformDataInteractor3D()
: m_Precision(6.5)
, m_InteractionMode(INTERACTION_MODE_TRANSLATION)
{
  m_OriginalGeometry = mitk::Geometry3D::New();

  // Initialize vector arithmetic
  m_ObjectNormal[0] = 0.0;
  m_ObjectNormal[1] = 0.0;
  m_ObjectNormal[2] = 1.0;

  m_CurrentRenderer = NULL;
  m_CurrentRenderWindow = NULL;
  m_CurrentRenderWindowInteractor = NULL;
  m_CurrentVtkRenderer = NULL;
  m_CurrentCamera = NULL;
  m_BoundingObjectNode = NULL;

  m_InteractionMode = false;
  m_AxesFixed = false;
}

mitk::AffineTransformDataInteractor3D::~AffineTransformDataInteractor3D()
{
}

void mitk::AffineTransformDataInteractor3D::SetInteractionMode(unsigned int interactionMode)
{
  m_InteractionMode = interactionMode;
}

void mitk::AffineTransformDataInteractor3D::SetInteractionModeToTranslation()
{
  m_InteractionMode = INTERACTION_MODE_TRANSLATION;
}

void mitk::AffineTransformDataInteractor3D::SetInteractionModeToRotation()
{
  m_InteractionMode = INTERACTION_MODE_ROTATION;
}

unsigned int mitk::AffineTransformDataInteractor3D::GetInteractionMode() const
{
  return m_InteractionMode;
}

void mitk::AffineTransformDataInteractor3D::SetPrecision(mitk::ScalarType precision)
{
  m_Precision = precision;
}

// Overwritten since this class can handle it better!
//float AffineTransformDataInteractor3D::CanHandleEvent(mitk::StateEvent const* stateEvent) const
//{
//  if (stateEvent->GetEvent() == NULL)
//    return 0.0;
//
//  if (this->GetCurrentState() != NULL && this->GetCurrentState()->GetTransition(stateEvent->GetId())!= NULL)
//    return 1.0;
//  else
//    return 0.0;
//
///*
//  int currentButtonState = 0;
//  int currentKeyPressed = 0;
//
//  //Handle MouseMove event
//  if (stateEvent->GetEvent()->GetType() == mitk::Type_MouseMove )
//  {
//    mitk::MouseEvent const *mouseEvent = dynamic_cast <const mitk::MouseEvent *> (stateEvent->GetEvent());
//    currentButtonState = mouseEvent->GetButtonState();
//    currentKeyPressed = mouseEvent->GetKey();
//    
//    m_currentButtonState = (int)(currentButtonState);
//    m_currentKeyPressed  = currentKeyPressed;
//
//    qDebug() <<"MouseEvent: " <<m_currentButtonState <<"key: " <<m_currentKeyPressed;
//
//    return 1.0;
//  }
//  else if (stateEvent->GetEvent()->GetType() == mitk::Type_KeyPress) 
//  {
//    mitk::KeyEvent const *keyEvent = dynamic_cast <const mitk::KeyEvent *> (stateEvent->GetEvent());
//    currentButtonState = keyEvent->GetButtonState();
//    currentKeyPressed = keyEvent->GetKey();
//    
//    m_currentButtonState = (int)(currentButtonState);
//    m_currentKeyPressed  = currentKeyPressed;
//
//    qDebug() <<"KeyEvent: " <<m_currentButtonState <<"key: " <<m_currentKeyPressed;
//
//    return 1.0;
//  }
//  else if (this->GetCurrentState()->GetTransition(stateEvent->GetId())!= NULL)
//    return 1.0;
//
//  return 0.0;
//*/
//}

bool mitk::AffineTransformDataInteractor3D::ColorizeSurface(vtkPolyData *polyData, const mitk::Point3D & /*pickedPoint*/, double scalar)
{
  if ( polyData == NULL )
  {
    return false;
  }

  //vtkPoints *points = polyData->GetPoints();
  vtkPointData *pointData = polyData->GetPointData();
  if ( pointData == NULL )
  {
    return false;
  }

  vtkDataArray *scalars = pointData->GetScalars();
  if ( scalars == NULL )
  {
    return false;
  }

  for ( int i = 0; i < pointData->GetNumberOfTuples(); ++i )
  {
    scalars->SetComponent( i, 0, scalar );
  }

  polyData->Modified();
  pointData->Update();

  return true;

//  // Get Event and extract renderer
//  const mitk::Event *eventE = stateEvent->GetEvent();
//  if (eventE == NULL)
//    return false;
//
//  //mitk::BaseRenderer *renderer = NULL;
//  vtkRenderWindow *renderWindow = NULL;
//  //vtkRenderWindowInteractor * renderWindowInteractor
//  vtkRenderer * currentVtkRenderer;
//  vtkCamera *camera = NULL;
//
//  renderer = eventE->GetSender();
//  if ( renderer != NULL )
//  {
//    renderWindow = renderer->GetRenderWindow();
//    if ( renderWindow != NULL )
//    {
//      renderWindowInteractor = renderWindow->GetInteractor();
//      if ( renderWindowInteractor != NULL )
//      {
//        currentVtkRenderer = renderWindowInteractor->GetInteractorStyle()->GetCurrentRenderer();
//        if ( currentVtkRenderer != NULL )
//        {
//          camera = currentVtkRenderer->GetActiveCamera();
//        }
//        else return false;
//      }
//      else return false;
//    }
//    else return false;
//  }
//  else return false;
//
//  //All went fine
//  return true;
}

bool mitk::AffineTransformDataInteractor3D::UpdateCurrentRendererPointers(const mitk::InteractionEvent * interactionEvent)
{
  // Get Event and extract renderer
  if (interactionEvent == NULL)
    return false;

  m_CurrentRenderer = interactionEvent->GetSender();
  if (m_CurrentRenderer != NULL )
  {
    m_CurrentRenderWindow = m_CurrentRenderer->GetRenderWindow();
    if (m_CurrentRenderWindow != NULL )
    {
      m_CurrentRenderWindowInteractor = m_CurrentRenderWindow->GetInteractor();
      if (m_CurrentRenderWindowInteractor != NULL )
      {
        m_CurrentVtkRenderer = m_CurrentRenderWindowInteractor->GetInteractorStyle()->GetCurrentRenderer();
        if (m_CurrentVtkRenderer != NULL)
        {
          m_CurrentCamera = m_CurrentVtkRenderer->GetActiveCamera();
        }
        else return false;
      }
      else return false;
    }
    else return false;
  }
  else return false;

  //All went fine
  return true;
}


bool mitk::AffineTransformDataInteractor3D::CheckObject(const InteractionEvent *interactionEvent)
{
  
  if (!UpdateCurrentRendererPointers(interactionEvent) || this->GetDataNode()->GetData() == NULL)
  {
    return false;
  }

  // Re-enable VTK interactor (may have been disabled previously)
  if (m_CurrentRenderWindowInteractor != NULL)
    m_CurrentRenderWindowInteractor->Enable();

  // Check if we have a DisplayPositionEvent
  const mitk::DisplayPositionEvent *dpe = dynamic_cast< const mitk::DisplayPositionEvent * >(interactionEvent);     
  if (dpe == NULL)
  {
    //Could not resolve current display position: go back to start state
    return false;
  }

  m_CurrentlyPickedWorldPoint = dpe->GetWorldPosition();
  m_CurrentlyPickedDisplayPoint = dpe->GetDisplayPosition();
  
  // Get the timestep to also support 3D+t
  int timeStep = 0;
  mitk::ScalarType timeInMS = 0.0;
  
  if (m_CurrentRenderer != NULL)
  {
    timeStep = m_CurrentRenderer->GetTimeStep(this->GetDataNode()->GetData());
    timeInMS = m_CurrentRenderer->GetTime();
  }

  mitk::BaseGeometry* geometry = this->GetDataNode()->GetData()->GetUpdatedTimeGeometry()->GetGeometryForTimeStep(timeStep);

  if (geometry->IsInside(m_CurrentlyPickedWorldPoint))
  {
    return true;
  }
  else
  {
    return false;
  }

  return true;
}

bool mitk::AffineTransformDataInteractor3D::SelectObject(StateMachineAction* action, InteractionEvent* interactionEvent)
{
  // Color object red
  this->GetDataNode()->SetColor( 1.0, 0.0, 0.0 );
  
  mitk::RenderingManager::Pointer renderManager = mitk::RenderingManager::GetInstance();
  renderManager->RequestUpdateAll();

  return true;
}

bool mitk::AffineTransformDataInteractor3D::DeselectObject(StateMachineAction* action, InteractionEvent* interactionEvent)
{
  // Color object white
  this->GetDataNode()->SetColor( 1.0, 1.0, 1.0 );
  
  mitk::RenderingManager::Pointer renderManager = mitk::RenderingManager::GetInstance();
  renderManager->RequestUpdateAll();

  return true;
}

bool mitk::AffineTransformDataInteractor3D::InitMove(StateMachineAction* action, InteractionEvent* interactionEvent)
{
  if (!UpdateCurrentRendererPointers(interactionEvent) || this->GetDataNode()->GetData() == NULL)
    return false;

  // Disable VTK interactor (may have been enabled previously)
  if (m_CurrentRenderWindowInteractor != NULL)
    m_CurrentRenderWindowInteractor->Disable();

  // Check if we have a DisplayPositionEvent
  const mitk::DisplayPositionEvent *dpe = dynamic_cast< const mitk::DisplayPositionEvent * >( interactionEvent);     
  if (dpe == NULL)
    return false;

  m_InitialPickedWorldPoint = m_CurrentlyPickedWorldPoint;
  m_InitialPickedDisplayPoint = m_CurrentlyPickedDisplayPoint;

  if (m_CurrentVtkRenderer != NULL)
  {
    vtkInteractorObserver::ComputeDisplayToWorld(
      m_CurrentVtkRenderer,
      m_InitialPickedDisplayPoint[0],
      m_InitialPickedDisplayPoint[1],
      0.0, //m_InitialInteractionPickedPoint[2],
      m_InitialPickedPointWorld );
  }

  // Get the timestep to also support 3D+t
  int timeStep = 0;
    
  if (m_CurrentRenderer != NULL)
    timeStep = m_CurrentRenderer->GetTimeStep(this->GetDataNode()->GetData());

  // Make deep copy of current Geometry3D of the plane
  this->GetDataNode()->GetData()->UpdateOutputInformation(); // make sure that the Geometry is up-to-date
  m_OriginalGeometry = dynamic_cast<mitk::BaseGeometry*>(this->GetDataNode()->GetData()->GetGeometry(timeStep)->Clone().GetPointer());

  return true;
}

bool mitk::AffineTransformDataInteractor3D::Move(StateMachineAction* action, InteractionEvent* interactionEvent)
{
  if (!UpdateCurrentRendererPointers(interactionEvent) || this->GetDataNode()->GetData() == NULL)
    return false;

  // Check if we have a DisplayPositionEvent
  const mitk::DisplayPositionEvent *dpe = dynamic_cast< const mitk::DisplayPositionEvent * >(interactionEvent);     
  
  if (dpe == NULL)
    return false;

  m_CurrentlyPickedWorldPoint   = dpe->GetWorldPosition();
  m_CurrentlyPickedDisplayPoint = dpe->GetDisplayPosition();

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

  if ( m_InteractionMode == INTERACTION_MODE_TRANSLATION )
  {
    mitk::Point3D origin = m_OriginalGeometry->GetOrigin();

    mitk::Vector3D transformedObjectNormal;
    this->GetDataNode()->GetData()->GetGeometry( timeStep )->IndexToWorld(m_ObjectNormal, transformedObjectNormal);

    if (m_AxesFixed == true)
      this->GetDataNode()->GetData()->GetGeometry( timeStep )->SetOrigin(origin + transformedObjectNormal * (interactionMove * transformedObjectNormal) );
    else
      this->GetDataNode()->GetData()->GetGeometry( timeStep )->SetOrigin(origin + interactionMove);
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

      double rotationAngle = 360.0 * sqrt(l2/(size[0]*size[0]+size[1]*size[1]));

      //qDebug() <<"RotAngle: " <<rotationAngle;

      // Use center of data bounding box as center of rotation
      //m_OriginalGeometry = m_DataNode->GetData()->GetGeometry(timeStep)->Clone().GetPointer();
      mitk::Point3D rotationCenter;
      rotationCenter = m_OriginalGeometry->GetCenter();

      //qDebug() <<"RotCenter: " <<rotationCenter[0] <<rotationCenter[1] <<rotationCenter[2];

      // Reset current Geometry3D to original state (pre-interaction) and
      // apply rotation

      mitk::RotationOperation op(mitk::OpROTATE, rotationCenter, rotationAxis, rotationAngle );
      mitk::BaseGeometry::Pointer newGeometry = dynamic_cast<mitk::BaseGeometry*>(this->GetDataNode()->GetData()->GetGeometry(timeStep)->Clone().GetPointer());

      if (newGeometry.IsNotNull())
      {
        newGeometry->mitk::BaseGeometry::ExecuteOperation( &op );
        mitk::TimeGeometry::Pointer timeGeometry = this->GetDataNode()->GetData()->GetTimeGeometry();
        bool succ = false;
        if (timeGeometry.IsNotNull() && timeGeometry->IsValidTimeStep(timeStep))
        {
          timeGeometry->SetTimeStepGeometry( newGeometry, timeStep );
          this->GetDataNode()->GetData()->Modified();
          this->GetDataNode()->Modified();
          this->GetDataNode()->Update();
        }
      }
    }
  }

  if (m_BoundingObjectNode != NULL)
  {
   static_cast<mitk::BoundingObject * >(m_BoundingObjectNode->GetData())->FitGeometry(this->GetDataNode()->GetData()->GetGeometry());
  }
  interactionEvent->GetSender()->GetRenderingManager()->RequestUpdateAll();
 
  return true;
}

bool mitk::AffineTransformDataInteractor3D::AcceptMove(StateMachineAction* action, InteractionEvent* interactionEvent)
{
  mitk::StateEvent * newStateEvent = NULL;

  // Get the timestep to also support 3D+t
  int timeStep = 0;

  if (m_CurrentRenderer != NULL)
    timeStep = m_CurrentRenderer->GetTimeStep(this->GetDataNode()->GetData());

  m_OriginalGeometry = dynamic_cast<mitk::BaseGeometry*>(this->GetDataNode()->GetData()->GetGeometry(timeStep)->Clone().GetPointer());

  emit transformReady();
  return true;
}

 void mitk::AffineTransformDataInteractor3D::SetAxesFixed(bool on, int which)
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

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//                                         CUSTOM VTK AXES ACTOR
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

mitk::CustomVTKAxesActor::CustomVTKAxesActor() 
  : vtkAxesActor() 
{
  //default: 0.25
  m_AxesLabelWidth = 0.1; 
  this->XAxisLabel->SetWidth(0.1);
  this->YAxisLabel->SetWidth(0.1);
  this->ZAxisLabel->SetWidth(0.1);

  //default: 0.1
  m_AxesLabelHeight = 0.04;
  this->XAxisLabel->SetHeight(0.05);
  this->YAxisLabel->SetHeight(0.05);
  this->ZAxisLabel->SetHeight(0.05);

  vtkTextProperty* tprop = this->XAxisLabel->GetCaptionTextProperty();
  tprop->ItalicOff();
  tprop->BoldOff();
  tprop->ShadowOn();
  this->XAxisLabel->SetCaptionTextProperty(tprop);

  tprop = this->YAxisLabel->GetCaptionTextProperty();
  tprop->ItalicOff();
  tprop->BoldOff();
  tprop->ShadowOn();
  this->YAxisLabel->SetCaptionTextProperty(tprop);

  tprop = this->ZAxisLabel->GetCaptionTextProperty();
  tprop->ItalicOff();
  tprop->BoldOff();
  tprop->ShadowOn();
  this->ZAxisLabel->SetCaptionTextProperty(tprop);
}
