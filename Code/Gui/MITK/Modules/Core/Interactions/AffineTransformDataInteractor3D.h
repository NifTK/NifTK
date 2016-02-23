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


#ifndef AffineTransformDataInteractor3D_h
#define AffineTransformDataInteractor3D_h

#include <mitkDataInteractor.h>
#include <mitkBaseRenderer.h>
//
#include <vtkPolyData.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkLegendScaleActor.h>
#include <vtkAxesActor.h>
#include <vtkCaptionActor2D.h>

#include <QApplication>

namespace mitk
{

/**
  * \brief Affine interaction with objects in 3D windows.
  *
  * NOTE: The interaction mechanism is similar to that of vtkPlaneWidget
  *
  * \ingroup Interaction
  */
class AffineTransformDataInteractor3D : public QObject, public mitk::DataInteractor
{
  Q_OBJECT

signals:
  void transformReady();

public:
  enum { INTERACTION_MODE_TRANSLATION, INTERACTION_MODE_ROTATION };

  mitkClassMacro(AffineTransformDataInteractor3D, mitk::DataInteractor)
  itkFactorylessNewMacro(Self)
  itkCloneMacro(Self)

  void SetInteractionMode(unsigned int interactionMode);
  void SetInteractionModeToTranslation();
  void SetInteractionModeToRotation();
  unsigned int GetInteractionMode() const;

  void SetAxesFixed(bool on, int which = 0);

  /** \brief Sets the amount of precision */
  void SetPrecision(mitk::ScalarType precision);

  /**
    * \brief calculates how good the data, this statemachine handles, is hit
    * by the event.
    *
    * overwritten, cause we don't look at the boundingbox, we look at each point
    */
  //virtual float CanHandleEvent(mitk::StateEvent const *stateEvent) const;
  //vtkRenderer * GetCurrentVTKRenderer() { return m_currentVtkRenderer; }

  inline void SetBoundingObjectNode(mitk::DataNode * bObj) {m_BoundingObjectNode = bObj;}

protected:
  /**
    * \brief Constructor with Param n for limited Set of Points
    *
    * if no n is set, then the number of points is unlimited*
    */
  AffineTransformDataInteractor3D();
  virtual ~AffineTransformDataInteractor3D();

  /**
  * Here actions strings from the loaded state machine pattern are mapped to functions of
  * the DataInteractor. These functions are called when an action from the state machine pattern is executed.
  */
  virtual void ConnectActionsAndFunctions() override;

  bool ColorizeSurface(vtkPolyData *polyData, const mitk::Point3D &pickedPoint, double scalar = 0.0);

  //************************************************************************************************************************************/
  bool GetCurrentRenderer(const mitk::InteractionEvent * interactionEvent, vtkRenderWindowInteractor * renderWindowInteractor,  mitk::BaseRenderer * renderer);
  bool UpdateCurrentRendererPointers(const mitk::InteractionEvent * interactionEvent);

  bool CheckObject(const InteractionEvent *interactionEvent);
  bool SelectObject(StateMachineAction* action, InteractionEvent* interactionEvent);
  bool DeselectObject(StateMachineAction* action, InteractionEvent* interactionEvent);
  bool InitMove(StateMachineAction* action, InteractionEvent* interactionEvent);
  bool Move(StateMachineAction* action, InteractionEvent* interactionEvent);
  bool AcceptMove(StateMachineAction* action, InteractionEvent* interactionEvent);

private:

  /** \brief to store the value of precision to pick a point */
  mitk::ScalarType           m_Precision;
  bool                       m_InteractionMode;
  bool                       m_AxesFixed;

  mitk::Point3D              m_InitialPickedWorldPoint;
  mitk::Point2D              m_InitialPickedDisplayPoint;
  double                     m_InitialPickedPointWorld[4];

  mitk::Point3D              m_CurrentlyPickedWorldPoint;
  mitk::Point2D              m_CurrentlyPickedDisplayPoint;
  double                     m_CurrentlyPickedPointWorld[4];

  mitk::BaseGeometry::Pointer  m_Geometry;

  mitk::BaseGeometry::Pointer  m_OriginalGeometry;

  mitk::Vector3D             m_ObjectNormal;
  
  mitk::BaseRenderer        * m_CurrentRenderer;
  vtkRenderWindow           * m_CurrentRenderWindow;
  vtkRenderWindowInteractor * m_CurrentRenderWindowInteractor;
  vtkRenderer               * m_CurrentVtkRenderer;
  vtkCamera                 * m_CurrentCamera;
  
  vtkLegendScaleActor       * m_LegendActor;
  vtkAxesActor              * m_AxesActor;
  mitk::DataNode            * m_BoundingObjectNode;
};

 class CustomVTKAxesActor : public vtkAxesActor
 {
 public:
   inline void SetAxisLabelWidth(double w) {this->XAxisLabel->SetWidth(w); this->YAxisLabel->SetWidth(w); this->ZAxisLabel->SetWidth(w); }
   inline double GetAxisLabelWidth() { return m_AxesLabelWidth;}
   inline void SetAxisLabelHeight(double h) {this->XAxisLabel->SetHeight(h); this->YAxisLabel->SetHeight(h); this->ZAxisLabel->SetHeight(h);}
   inline double GetAxisLabelHeight() {return m_AxesLabelHeight;}

   CustomVTKAxesActor();

 private:
   double m_AxesLabelWidth;
   double m_AxesLabelHeight;
 };

 }

#endif /* MITKAFFINEDATAINTERACTOR3D_H*/
