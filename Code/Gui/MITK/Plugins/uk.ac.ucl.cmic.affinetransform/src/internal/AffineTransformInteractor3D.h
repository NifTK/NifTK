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


#ifndef AffineTransformInteractor3D_h
#define AffineTransformInteractor3D_h

#include <mitkInteractor.h>
#include <mitkInteractionConst.h>
#include <mitkCommon.h>
#include <mitkSurface.h>
#include <mitkBaseRenderer.h>
#include <mitkBoundingObject.h>

#include <vtkPolyData.h>
#include <vtkType.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkInteractorStyle.h>
#include <vtkLegendScaleActor.h>
#include <vtkAxesActor.h>
#include <vtkActor.h>
#include <vtkCaptionActor2D.h>
#include <vtkObject.h>
#include <vtkObjectFactory.h>
#include <vtkSmartPointer.h>
#include <vtkTextProperty.h>

#include <QApplication>

/**
  * \brief Affine interaction with objects in 3D windows.
  *
  * NOTE: The interaction mechanism is similar to that of vtkPlaneWidget
  *
  * \ingroup Interaction
  */
class AffineTransformInteractor3D : public QObject, public mitk::Interactor
{
  Q_OBJECT

signals:
  void transformReady();

public:
  enum { INTERACTION_MODE_TRANSLATION, INTERACTION_MODE_ROTATION };

  mitkClassMacro(AffineTransformInteractor3D, mitk::Interactor);
  mitkNewMacro3Param(Self, const char *, mitk::DataNode *, int);
  mitkNewMacro2Param(Self, const char *, mitk::DataNode *);

  void SetInteractionMode( unsigned int interactionMode );
  void SetInteractionModeToTranslation();
  void SetInteractionModeToRotation();
  unsigned int GetInteractionMode() const;

  void SetAxesFixed(bool on, int which = 0);

  /** \brief Sets the amount of precision */
  void SetPrecision(mitk::ScalarType precision );

  /**
    * \brief calculates how good the data, this statemachine handles, is hit
    * by the event.
    *
    * overwritten, cause we don't look at the boundingbox, we look at each point
    */
  virtual float CanHandleEvent(mitk::StateEvent const *stateEvent) const;
  //vtkRenderer * GetCurrentVTKRenderer() { return m_currentVtkRenderer; }

  inline void SetBoundingObjectNode(mitk::DataNode * bObj) { m_boundingObjectNode = bObj; }

protected:
  /**
    * \brief Constructor with Param n for limited Set of Points
    *
    * if no n is set, then the number of points is unlimited*
    */
  AffineTransformInteractor3D(const char *type, mitk::DataNode *dataNode, int n = -1);
  virtual ~AffineTransformInteractor3D();

  bool ColorizeSurface( vtkPolyData *polyData, const mitk::Point3D &pickedPoint, double scalar = 0.0 );

  //************************************************************************************************************************************/
  bool GetCurrentRenderer(const mitk::StateEvent * event, vtkRenderWindowInteractor * renderWindowInteractor,  mitk::BaseRenderer * renderer);
  bool UpdateCurrentRendererPointers(const mitk::StateEvent * stateEvent);

  bool OnAcCheckObject(mitk::Action * action, const mitk::StateEvent * stateEvent);
  bool OnAcSelectPickedObject(mitk::Action * action, const mitk::StateEvent * stateEvent);
  bool OnAcDeselectPickedObject(mitk::Action * action, const mitk::StateEvent * stateEvent);
  bool OnAcInitMove(mitk::Action * action, const mitk::StateEvent * stateEvent);
  bool OnAcMove(mitk::Action * action, const mitk::StateEvent * stateEvent);
  bool OnAcAccept(mitk::Action * action, const mitk::StateEvent * stateEvent);

private:

  /** \brief to store the value of precision to pick a point */
  mitk::ScalarType           m_Precision;
  bool                       m_InteractionMode;
  bool                       m_AxesFixed;

  mitk::Point3D              m_InitialPickedWorldPoint;
  mitk::Point2D              m_InitialPickedDisplayPoint;
  vtkFloatingPointType       m_InitialPickedPointWorld[4];

  mitk::Point3D              m_CurrentlyPickedWorldPoint;
  mitk::Point2D              m_CurrentlyPickedDisplayPoint;
  vtkFloatingPointType       m_CurrentlyPickedPointWorld[4];

  mitk::Geometry3D::Pointer  m_Geometry;

  mitk::Geometry3D::Pointer  m_OriginalGeometry;
  //mitk::SlicedGeometry3D::Pointer m_OriginalTimeSlicedGeometry;

  mitk::Vector3D             m_ObjectNormal;
  
  mitk::BaseRenderer        * m_currentRenderer;
  vtkRenderWindow           * m_currentRenderWindow;
  vtkRenderWindowInteractor * m_currentRenderWindowInteractor;
  vtkRenderer               * m_currentVtkRenderer;
  vtkCamera                 * m_currentCamera;
  
  vtkLegendScaleActor       * m_legendActor;
  vtkAxesActor              * m_axesActor;
  mitk::DataNode            * m_boundingObjectNode;
};

class CustomVTKAxesActor : public vtkAxesActor
{
public:
  inline void SetAxisLabelWidth(double w) { this->XAxisLabel->SetWidth(w); this->YAxisLabel->SetWidth(w); this->ZAxisLabel->SetWidth(w); }
  inline double GetAxisLabelWidth() { return m_axesLabelWidth;}
  inline void SetAxisLabelHeight(double h) { this->XAxisLabel->SetHeight(h); this->YAxisLabel->SetHeight(h); this->ZAxisLabel->SetHeight(h);}
  inline double GetAxisLabelHeight() { return m_axesLabelHeight;}

  CustomVTKAxesActor();
//  virtual ~CustomVTKAxesActor();

private:
  double m_axesLabelWidth;
  double m_axesLabelHeight;
};

#endif /* MITKAFFINEINTERACTOR3D_H_HEADER_INCLUDED */
