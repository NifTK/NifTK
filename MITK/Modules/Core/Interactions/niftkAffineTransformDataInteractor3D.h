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


#ifndef niftkAffineTransformDataInteractor3D_h
#define niftkAffineTransformDataInteractor3D_h

#include "niftkCoreExports.h"

#include <mitkDataInteractor.h>
#include <mitkBaseRenderer.h>
//
#include <vtkPolyData.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkLegendScaleActor.h>
#include <vtkAxesActor.h>

#include <QApplication>

namespace niftk
{

/**
  * \brief Affine data interaction with objects in 3D windows.
  *
  * \ingroup Interaction
  */
class NIFTKCORE_EXPORT AffineTransformDataInteractor3D : public QObject, public mitk::DataInteractor
{
  Q_OBJECT

signals:
  void transformReady();

public:
  enum { INTERACTION_MODE_TRANSLATION, INTERACTION_MODE_ROTATION };

  mitkClassMacro(AffineTransformDataInteractor3D, mitk::DataInteractor)
  itkFactorylessNewMacro(Self)
  itkCloneMacro(Self)

  /** \brief Set interaction mode by enum. */
  void SetInteractionMode(unsigned int interactionMode);
  /** \brief Set interaction mode to translate. */
  void SetInteractionModeToTranslation();
  /** \brief Set interaction mode to rotate. */
  void SetInteractionModeToRotation();
  /** \brief Return the enum of the set interaction mode*/
  unsigned int GetInteractionMode() const;

  /** \brief Sets if an axis should be fit and which axis*/
  void SetAxesFixed(bool on, int which = 0);

  /** \brief Sets to node to be used for the bounding box.*/
  inline void SetBoundingObjectNode(mitk::DataNode * bObj) {m_BoundingObjectNode = bObj;}

  vtkMatrix4x4* GetUpdatedGeometry();

protected:

  AffineTransformDataInteractor3D();
  virtual ~AffineTransformDataInteractor3D();

  /**
  * Here actions strings from the loaded state machine pattern are mapped to functions of
  * the DataInteractor. These functions are called when an action from the state machine pattern is executed.
  */
  virtual void ConnectActionsAndFunctions() override;

  bool UpdateCurrentRendererPointers(const mitk::InteractionEvent * interactionEvent);

  bool CheckObject(const mitk::InteractionEvent *interactionEvent);

  bool SelectObject(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent);

  bool DeselectObject(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent);

  bool InitMove(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent);

  bool Move(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent);

  bool AcceptMove(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent);

private:

  /** \brief to store the value of precision to pick a point */
  bool                       m_InteractionMode;
  bool                       m_AxesFixed;

  mitk::Point2D              m_InitialPickedDisplayPoint;
  double                     m_InitialPickedPointWorld[4];

  mitk::Point2D              m_CurrentlyPickedDisplayPoint;
  double                     m_CurrentlyPickedPointWorld[4];

  mitk::BaseGeometry::Pointer  m_OriginalGeometry;

  mitk::BaseGeometry::Pointer  m_UpdatedGeometry;

  mitk::Vector3D               m_ObjectNormal;

  mitk::BaseRenderer        * m_CurrentRenderer;
  vtkRenderer               * m_CurrentVtkRenderer;
  vtkCamera                 * m_CurrentCamera;

  mitk::DataNode            * m_BoundingObjectNode;
};

}

#endif /* MITKAFFINEDATAINTERACTOR3D_H*/
