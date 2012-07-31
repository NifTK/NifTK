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


#include "AffineTransformInteractor3D.h"

#include "mitkInteractionConst.h"
#include "mitkPointOperation.h"
#include "mitkPositionEvent.h"
#include "mitkStatusBar.h"
#include "mitkDataNode.h"
#include "mitkInteractionConst.h"
#include "mitkAction.h"
#include "mitkStateMachine.h"
#include "mitkStateEvent.h"
#include "mitkOperationEvent.h"
#include "mitkUndoController.h"
#include "mitkStateMachineFactory.h"
#include "mitkStateTransitionOperation.h"
#include "mitkRenderingManager.h"
#include "mitkRotationOperation.h"

#include <vtkCamera.h>
#include <vtkPoints.h>
#include <vtkPointData.h>
#include <vtkDataArray.h>

#include <QDebug>


//how precise must the user pick the point
//default value
AffineTransformInteractor3D
::AffineTransformInteractor3D(const char * type, mitk::DataNode* dataNode, int /* n */ )
: mitk::Interactor(type, dataNode),
  m_Precision(6.5),
  m_InteractionMode(INTERACTION_MODE_TRANSLATION)
{
  m_OriginalGeometry = mitk::Geometry3D::New();

  // Initialize vector arithmetic
  m_ObjectNormal[0] = 0.0;
  m_ObjectNormal[1] = 0.0;
  m_ObjectNormal[2] = 1.0;

  m_currentRenderer = NULL;
  m_currentRenderWindow = NULL;
  m_currentRenderWindowInteractor = NULL;
  m_currentVtkRenderer = NULL;
  m_currentCamera = NULL;
  m_boundingObjectNode = NULL;

  m_InteractionMode = false;
  m_AxesFixed = false;

  mitk::StateMachine::AddActionFunction(mitk::AcCHECKOBJECT, new mitk::TSpecificStateMachineFunctor<Self>(this, &Self::OnAcCheckObject));
  mitk::StateMachine::AddActionFunction(mitk::AcSELECTPICKEDOBJECT, new mitk::TSpecificStateMachineFunctor<Self>(this, &Self::OnAcSelectPickedObject));
  mitk::StateMachine::AddActionFunction(mitk::AcDESELECTOBJECT, new mitk::TSpecificStateMachineFunctor<Self>(this, &Self::OnAcDeselectPickedObject));
  mitk::StateMachine::AddActionFunction(mitk::AcINITMOVE, new mitk::TSpecificStateMachineFunctor<Self>(this, &Self::OnAcInitMove));
  mitk::StateMachine::AddActionFunction(mitk::AcMOVE, new mitk::TSpecificStateMachineFunctor<Self>(this, &Self::OnAcMove));
  mitk::StateMachine::AddActionFunction(mitk::AcACCEPT, new mitk::TSpecificStateMachineFunctor<Self>(this, &Self::OnAcAccept));    
}

AffineTransformInteractor3D::~AffineTransformInteractor3D()
{
}

void AffineTransformInteractor3D::SetInteractionMode( unsigned int interactionMode )
{
  m_InteractionMode = interactionMode;
}

void AffineTransformInteractor3D::SetInteractionModeToTranslation()
{
  m_InteractionMode = INTERACTION_MODE_TRANSLATION;
}

void AffineTransformInteractor3D::SetInteractionModeToRotation()
{
  m_InteractionMode = INTERACTION_MODE_ROTATION;
}


unsigned int AffineTransformInteractor3D::GetInteractionMode() const
{
  return m_InteractionMode;
}


void AffineTransformInteractor3D::SetPrecision( mitk::ScalarType precision )
{
  m_Precision = precision;
}

// Overwritten since this class can handle it better!
float AffineTransformInteractor3D::CanHandleEvent(mitk::StateEvent const* stateEvent) const
{
  if (stateEvent->GetEvent() == NULL)
    return 0.0;

  if (this->GetCurrentState() != NULL && this->GetCurrentState()->GetTransition(stateEvent->GetId())!= NULL)
    return 1.0;
  else
    return 0.0;

/*
  int currentButtonState = 0;
  int currentKeyPressed = 0;

  //Handle MouseMove event
  if (stateEvent->GetEvent()->GetType() == mitk::Type_MouseMove )
  {
    mitk::MouseEvent const *mouseEvent = dynamic_cast <const mitk::MouseEvent *> (stateEvent->GetEvent());
    currentButtonState = mouseEvent->GetButtonState();
    currentKeyPressed = mouseEvent->GetKey();
    
    m_currentButtonState = (int)(currentButtonState);
    m_currentKeyPressed  = currentKeyPressed;

    qDebug() <<"MouseEvent: " <<m_currentButtonState <<"key: " <<m_currentKeyPressed;

    return 1.0;
  }
  else if (stateEvent->GetEvent()->GetType() == mitk::Type_KeyPress) 
  {
    mitk::KeyEvent const *keyEvent = dynamic_cast <const mitk::KeyEvent *> (stateEvent->GetEvent());
    currentButtonState = keyEvent->GetButtonState();
    currentKeyPressed = keyEvent->GetKey();
    
    m_currentButtonState = (int)(currentButtonState);
    m_currentKeyPressed  = currentKeyPressed;

    qDebug() <<"KeyEvent: " <<m_currentButtonState <<"key: " <<m_currentKeyPressed;

    return 1.0;
  }
  else if (this->GetCurrentState()->GetTransition(stateEvent->GetId())!= NULL)
    return 1.0;

  return 0.0;
*/
}
bool AffineTransformInteractor3D::ColorizeSurface( vtkPolyData *polyData, const mitk::Point3D & /*pickedPoint*/, double scalar)
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

  for ( unsigned int i = 0; i < pointData->GetNumberOfTuples(); ++i )
  {
    scalars->SetComponent( i, 0, scalar );
  }

  polyData->Modified();
  pointData->Update();

  return true;
}

//*****************************************************************************************************//
 
//bool AffineTransformInteractor3D::GetCurrentRenderer(const mitk::StateEvent * stateEvent, vtkRenderWindowInteractor * renderWindowInteractor, mitk::BaseRenderer * renderer)
//{
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
//}

bool AffineTransformInteractor3D::UpdateCurrentRendererPointers(const mitk::StateEvent * stateEvent)
{
  // Get Event and extract renderer
  const mitk::Event *eventE = stateEvent->GetEvent();
  if (eventE == NULL)
    return false;

  m_currentRenderer = eventE->GetSender();
  if (m_currentRenderer != NULL )
  {
    m_currentRenderWindow = m_currentRenderer->GetRenderWindow();
    if (m_currentRenderWindow != NULL )
    {
      m_currentRenderWindowInteractor = m_currentRenderWindow->GetInteractor();
      if (m_currentRenderWindowInteractor != NULL )
      {
        m_currentVtkRenderer = m_currentRenderWindowInteractor->GetInteractorStyle()->GetCurrentRenderer();
        if (m_currentVtkRenderer != NULL)
        {
          m_currentCamera = m_currentVtkRenderer->GetActiveCamera();
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


bool AffineTransformInteractor3D::OnAcCheckObject(mitk::Action* action, const mitk::StateEvent* stateEvent)
{
  mitk::StateEvent * newStateEvent = NULL;
  
  if (!UpdateCurrentRendererPointers(stateEvent) || m_DataNode->GetData() == NULL)
  {
    //Renderer not ready for interaction: go back to start state
    newStateEvent = new mitk::StateEvent(mitk::EIDNO);
    this->HandleEvent(newStateEvent);
    return false;
  }

  // Re-enable VTK interactor (may have been disabled previously)
  if (m_currentRenderWindowInteractor != NULL)
    m_currentRenderWindowInteractor->Enable();

  // Check if we have a DisplayPositionEvent
  const mitk::DisplayPositionEvent *dpe = dynamic_cast< const mitk::DisplayPositionEvent * >( stateEvent->GetEvent() );     
  if (dpe == NULL)
  {
    //Could not resolve current display position: go back to start state
    newStateEvent = new mitk::StateEvent(mitk::EIDNO);
    this->HandleEvent(newStateEvent);
    return false;
  }

  m_CurrentlyPickedWorldPoint = dpe->GetWorldPosition();
  m_CurrentlyPickedDisplayPoint = dpe->GetDisplayPosition();
  
  // Get the timestep to also support 3D+t
  int timeStep = 0;
  mitk::ScalarType timeInMS = 0.0;
  
  if (m_currentRenderer != NULL)
  {
    timeStep = m_currentRenderer->GetTimeStep(m_DataNode->GetData());
    timeInMS = m_currentRenderer->GetTime();
  }

  //qDebug() <<"WorldPoint: " <<m_CurrentlyPickedWorldPoint.operator [](0) <<m_CurrentlyPickedWorldPoint.operator [](1) <<m_CurrentlyPickedWorldPoint.operator [](2);
  //qDebug() <<"DisplayPos: " <<m_CurrentlyPickedDisplayPoint.operator [](0) <<m_CurrentlyPickedDisplayPoint.operator [](1) <<m_CurrentlyPickedDisplayPoint.operator [](2);
  
  mitk::Geometry3D * geometry = m_DataNode->GetData()->GetUpdatedTimeSlicedGeometry()->GetGeometry3D(timeStep);

  if (geometry->IsInside(m_CurrentlyPickedWorldPoint))
  {
    //qDebug() <<"Current 3D position is within the selected object's bounding box" ;
    newStateEvent = new mitk::StateEvent(mitk::EIDYES );
    this->HandleEvent( newStateEvent );
    return true;
  }
  else
  {
    //qDebug() <<"Current 3D position is out of the selected object's bounding box" ;
    newStateEvent = new mitk::StateEvent(mitk::EIDNO );
    this->HandleEvent( newStateEvent );
    return false;
  }

  return true;
}

bool AffineTransformInteractor3D::OnAcSelectPickedObject(mitk::Action * action, const mitk::StateEvent * stateEvent)
{
  // Color object red
  m_DataNode->SetColor( 1.0, 0.0, 0.0 );
  mitk::RenderingManager::GetInstance()->RequestUpdateAll();

  return true;
}

bool AffineTransformInteractor3D::OnAcDeselectPickedObject(mitk::Action * action, const mitk::StateEvent * stateEvent)
{
  // Color object white
  m_DataNode->SetColor( 1.0, 1.0, 1.0 );
  mitk::RenderingManager::GetInstance()->RequestUpdateAll();

  return true;
}

bool AffineTransformInteractor3D::OnAcInitMove(mitk::Action * action, const mitk::StateEvent * stateEvent)
{
  if (!UpdateCurrentRendererPointers(stateEvent) || m_DataNode->GetData() == NULL)
    return false;

  // Disable VTK interactor (may have been enabled previously)
  if (m_currentRenderWindowInteractor != NULL)
    m_currentRenderWindowInteractor->Disable();

  // Check if we have a DisplayPositionEvent
  const mitk::DisplayPositionEvent *dpe = dynamic_cast< const mitk::DisplayPositionEvent * >( stateEvent->GetEvent() );     
  if (dpe == NULL)
    return false;

  m_InitialPickedWorldPoint = m_CurrentlyPickedWorldPoint;
  m_InitialPickedDisplayPoint = m_CurrentlyPickedDisplayPoint;

  if (m_currentVtkRenderer != NULL)
  {
    vtkInteractorObserver::ComputeDisplayToWorld(
      m_currentVtkRenderer,
      m_InitialPickedDisplayPoint[0],
      m_InitialPickedDisplayPoint[1],
      0.0, //m_InitialInteractionPickedPoint[2],
      m_InitialPickedPointWorld );
  }

  // Get the timestep to also support 3D+t
  int timeStep = 0;
    
  if (m_currentRenderer != NULL)
    timeStep = m_currentRenderer->GetTimeStep(m_DataNode->GetData());

  // Make deep copy of current Geometry3D of the plane
  m_DataNode->GetData()->UpdateOutputInformation(); // make sure that the Geometry is up-to-date
  m_OriginalGeometry = static_cast<mitk::Geometry3D * >(m_DataNode->GetData()->GetGeometry(timeStep)->Clone().GetPointer());

  return true;
}

bool AffineTransformInteractor3D::OnAcMove(mitk::Action * action, const mitk::StateEvent * stateEvent)
{
  if (!UpdateCurrentRendererPointers(stateEvent) || m_DataNode->GetData() == NULL)
    return false;

  // Check if we have a DisplayPositionEvent
  const mitk::DisplayPositionEvent *dpe = dynamic_cast< const mitk::DisplayPositionEvent * >( stateEvent->GetEvent() );     
  
  if (dpe == NULL)
    return false;

  m_CurrentlyPickedWorldPoint   = dpe->GetWorldPosition();
  m_CurrentlyPickedDisplayPoint = dpe->GetDisplayPosition();

  mitk::Vector3D interactionMove;

  if (m_currentVtkRenderer != NULL)
  {
    vtkInteractorObserver::ComputeDisplayToWorld(
      m_currentVtkRenderer,
      m_CurrentlyPickedDisplayPoint[0],
      m_CurrentlyPickedDisplayPoint[1],
      0.0, //m_InitialInteractionPickedPoint[2],
      m_CurrentlyPickedPointWorld);
  }
  interactionMove[0] = m_CurrentlyPickedPointWorld[0] - m_InitialPickedPointWorld[0];
  interactionMove[1] = m_CurrentlyPickedPointWorld[1] - m_InitialPickedPointWorld[1];
  interactionMove[2] = m_CurrentlyPickedPointWorld[2] - m_InitialPickedPointWorld[2];

  //qDebug() <<"InteractionMove: " <<interactionMove[0] <<interactionMove[1] <<interactionMove[2];
  //interactionMove[0] = m_CurrentlyPickedWorldPoint[0] - m_InitialPickedWorldPoint[0];
  //interactionMove[1] = m_CurrentlyPickedWorldPoint[1] - m_InitialPickedWorldPoint[1];
  //interactionMove[2] = m_CurrentlyPickedWorldPoint[2] - m_InitialPickedWorldPoint[2];

  // Get the timestep to also support 3D+t
  int timeStep = 0;
    
  if (m_currentRenderer != NULL)
    timeStep = m_currentRenderer->GetTimeStep(m_DataNode->GetData());

  if ( m_InteractionMode == INTERACTION_MODE_TRANSLATION )
  {
    mitk::Point3D origin = m_OriginalGeometry->GetOrigin();

    mitk::Vector3D transformedObjectNormal;
    m_DataNode->GetData()->GetGeometry( timeStep )->IndexToWorld(m_ObjectNormal, transformedObjectNormal);

    if (m_AxesFixed == true)
      m_DataNode->GetData()->GetGeometry( timeStep )->SetOrigin(origin + transformedObjectNormal * (interactionMove * transformedObjectNormal) );
    else
      m_DataNode->GetData()->GetGeometry( timeStep )->SetOrigin(origin + interactionMove);
  }
  else if (m_InteractionMode == INTERACTION_MODE_ROTATION)
  {
    if (m_currentCamera != NULL)
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
        vtkFloatingPointType vpn[3];
        m_currentCamera->GetViewPlaneNormal( vpn );

        mitk::Vector3D viewPlaneNormal;
        viewPlaneNormal[0] = vpn[0];
        viewPlaneNormal[1] = vpn[1];
        viewPlaneNormal[2] = vpn[2];

        //qDebug() <<"PlaneNormal: " <<vpn[0] <<vpn[1] <<vpn[2];

        rotationAxis = itk::CrossProduct(viewPlaneNormal, interactionMove);
        rotationAxis.Normalize();
      }

      //qDebug() <<"RotAxis: " <<rotationAxis[0] <<rotationAxis[1] <<rotationAxis[2];

      int *size = m_currentVtkRenderer->GetSize();
      double l2 =
        (m_CurrentlyPickedDisplayPoint[0] - m_InitialPickedDisplayPoint[0]) *
        (m_CurrentlyPickedDisplayPoint[0] - m_InitialPickedDisplayPoint[0]) +
        (m_CurrentlyPickedDisplayPoint[1] - m_InitialPickedDisplayPoint[1]) *
        (m_CurrentlyPickedDisplayPoint[1] - m_InitialPickedDisplayPoint[1]);

      double rotationAngle = 360.0 * sqrt(l2/(size[0]*size[0]+size[1]*size[1]));

      //qDebug() <<"RotAngle: " <<rotationAngle;

      // Use center of data bounding box as center of rotation
      //m_OriginalGeometry = static_cast<mitk::Geometry3D * >(m_DataNode->GetData()->GetGeometry(timeStep)->Clone().GetPointer());
      mitk::Point3D rotationCenter;
      rotationCenter = m_OriginalGeometry->GetCenter();

      //qDebug() <<"RotCenter: " <<rotationCenter[0] <<rotationCenter[1] <<rotationCenter[2];

      // Reset current Geometry3D to original state (pre-interaction) and
      // apply rotation
      
      mitk::RotationOperation op(mitk::OpROTATE, rotationCenter, rotationAxis, rotationAngle );
      mitk::Geometry3D::Pointer newGeometry = static_cast<mitk::Geometry3D * >(m_DataNode->GetData()->GetGeometry(timeStep)->Clone().GetPointer());

      if (newGeometry.IsNotNull())
      {
        newGeometry->mitk::Geometry3D::ExecuteOperation( &op );
        mitk::TimeSlicedGeometry::Pointer timeSlicedGeometry = m_DataNode->GetData()->GetTimeSlicedGeometry();
        bool succ = false;
        if (timeSlicedGeometry.IsNotNull())
        {
          succ = timeSlicedGeometry->SetGeometry3D( newGeometry, timeStep );
          
          if (succ)
          {
            m_DataNode->GetData()->Modified();
            m_DataNode->Modified();
            m_DataNode->Update();
          }

        }
      }
    }
  }
  
  if (m_boundingObjectNode != NULL)
  {
   static_cast<mitk::BoundingObject * >(m_boundingObjectNode->GetData())->FitGeometry(m_DataNode->GetData()->GetTimeSlicedGeometry());
  }
  mitk::RenderingManager::GetInstance()->RequestUpdateAll();
 
  return true;
}

bool AffineTransformInteractor3D::OnAcAccept(mitk::Action * action, const mitk::StateEvent * stateEvent)
{
  mitk::StateEvent * newStateEvent = NULL;
  
  // Get the timestep to also support 3D+t
  int timeStep = 0;
    
  if (m_currentRenderer != NULL)
    timeStep = m_currentRenderer->GetTimeStep(m_DataNode->GetData());

  m_OriginalGeometry = static_cast<mitk::Geometry3D * >(m_DataNode->GetData()->GetGeometry(timeStep)->Clone().GetPointer());
    
  emit transformReady();
  newStateEvent = new mitk::StateEvent(mitk::EIDYES );
  this->HandleEvent( newStateEvent );
  return true;
}

 void AffineTransformInteractor3D::SetAxesFixed(bool on, int which)
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

CustomVTKAxesActor::CustomVTKAxesActor() 
  : vtkAxesActor() 
{ 
  //default: 0.25
  m_axesLabelWidth = 0.1; 
  this->XAxisLabel->SetWidth(0.1);
  this->YAxisLabel->SetWidth(0.1);
  this->ZAxisLabel->SetWidth(0.1);

  //default: 0.1
  m_axesLabelHeight = 0.04;
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
