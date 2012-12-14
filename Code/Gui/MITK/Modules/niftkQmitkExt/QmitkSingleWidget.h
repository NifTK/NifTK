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

#ifndef QMITKSINGLEWIDGET_H_
#define QMITKSINGLEWIDGET_H_

#include "niftkQmitkExtExports.h"

#include "mitkPositionTracker.h"
#include "mitkSlicesRotator.h"
#include "mitkSlicesSwiveller.h"
#include "mitkRenderWindowFrame.h"
#include "mitkManufacturerLogo.h"
#include "mitkGradientBackground.h"
#include "mitkCoordinateSupplier.h"
#include "mitkDataStorage.h"

#include "mitkMouseModeSwitcher.h"

#include <qwidget.h>
#include <qsplitter.h>
#include <QFrame>

#include <QmitkRenderWindow.h>
#include <QmitkLevelWindowWidget.h>

#include "vtkTextProperty.h"
#include "vtkCornerAnnotation.h"

class QHBoxLayout;
class QVBoxLayout;
class QGridLayout;
class QSpacerItem;
class QmitkLevelWindowWidget;
class QmitkRenderWindow;

namespace mitk {
class RenderingManager;
}

class NIFTKQMITKEXT_EXPORT QmitkSingleWidget : public QWidget
{
  Q_OBJECT

public:

  QmitkSingleWidget(QWidget* parent = 0, Qt::WindowFlags f = 0, mitk::RenderingManager* renderingManager = 0);
  virtual ~QmitkSingleWidget();

  mitk::SliceNavigationController*
  GetTimeNavigationController();

  void RequestUpdate();

  void ForceImmediateUpdate();

  mitk::MouseModeSwitcher* GetMouseModeSwitcher();


  QmitkRenderWindow* GetRenderWindow1() const;

  const mitk::Point3D &
  GetLastLeftClickPosition() const;

  const mitk::Point3D
  GetCrossPosition() const;

  void EnablePositionTracking();

  void DisablePositionTracking();

  int GetLayout() const;

  mitk::SlicesRotator * GetSlicesRotator() const;

  mitk::SlicesSwiveller * GetSlicesSwiveller() const;

  bool GetGradientBackgroundFlag() const;

  /*!
  \brief Access node of widget plane 1
  \return DataNode holding widget plane 1
  */
  mitk::DataNode::Pointer GetWidgetPlane1();
  /*!
  \brief Access node of widget plane 2
  \return DataNode holding widget plane 2
  */
  mitk::DataNode::Pointer GetWidgetPlane2();
  /*!
  \brief Access node of widget plane 3
  \return DataNode holding widget plane 3
  */
  mitk::DataNode::Pointer GetWidgetPlane3();
  /*!
  \brief Convenience method to access node of widget planes
  \param id number of widget plane to be returned
  \return DataNode holding widget plane 3
  */
  mitk::DataNode::Pointer GetWidgetPlane(int id);

  bool IsColoredRectanglesEnabled() const;

  bool IsDepartmentLogoEnabled() const;

  bool IsCrosshairNavigationEnabled() const;

  void InitializeWidget();

  /// called when the SingleWidget is closed to remove the 3 widget planes and the helper node from the DataStorage
  void RemoveCameraFromDataStorage();

  void AddCameraToDataStorage();

  void SetDataStorage( mitk::DataStorage* ds );

  /** \brief Listener to the CrosshairPositionEvent

    Ensures the CrosshairPositionEvent is handled only once and at the end of the Qt-Event loop
  */
  void HandleCrosshairPositionEvent();

  /// activate Menu Widget. true: activated, false: deactivated
  void ActivateMenuWidget( bool state );

  bool IsMenuWidgetEnabled() const;
  
protected:

  void UpdateAllWidgets();

  void HideAllWidgetToolbars();

public slots:

  /// Receives the signal from HandleCrosshairPositionEvent, executes the StatusBar update
  void HandleCrosshairPositionEventDelayed();

<<<<<<< HEAD
=======

>>>>>>> 50812c6a718278a1cec11251807b9c6988155e08
  void changeLayoutToDefault();

  void changeLayoutToBig3D();

<<<<<<< HEAD
=======
 


>>>>>>> 50812c6a718278a1cec11251807b9c6988155e08
  void Fit();

  void InitPositionTracking();

  void AddDisplayPlaneSubTree();

  void EnableStandardLevelWindow();

  void DisableStandardLevelWindow();

  bool InitializeStandardViews( const mitk::Geometry3D * geometry );

  void wheelEvent( QWheelEvent * e );

  void mousePressEvent(QMouseEvent * e);

  void moveEvent( QMoveEvent* e );

  void leaveEvent ( QEvent * e  );

  void EnsureDisplayContainsPoint(
    mitk::DisplayGeometry* displayGeometry, const mitk::Point3D& p);

  void MoveCrossToPosition(const mitk::Point3D& newPosition);

  void EnableNavigationControllerEventListening();

  void DisableNavigationControllerEventListening();

  void EnableGradientBackground();

  void DisableGradientBackground();

  void EnableDepartmentLogo();

  void DisableDepartmentLogo();
  
  void EnableColoredRectangles();

  void DisableColoredRectangles();

  void SetWidgetPlaneVisibility(const char* widgetName, bool visible, mitk::BaseRenderer *renderer=NULL);

  void SetWidgetPlanesVisibility(bool visible, mitk::BaseRenderer *renderer=NULL);

  void SetWidgetPlanesLocked(bool locked);

  void SetWidgetPlanesRotationLocked(bool locked);

  void SetWidgetPlanesRotationLinked( bool link );

  void SetWidgetPlaneMode( int mode );

  void SetGradientBackgroundColors( const mitk::Color & upper, const mitk::Color & lower );

  void SetDepartmentLogoPath( const char * path );

  void SetWidgetPlaneModeToSlicing( bool activate );

  void SetWidgetPlaneModeToRotation( bool activate );

  void SetWidgetPlaneModeToSwivel( bool activate );

  void OnLayoutDesignChanged( int layoutDesignIndex );

  void ResetCrosshair();

  void MouseModeSelected( mitk::MouseModeSwitcher::MouseMode mouseMode );


signals:

  void LeftMouseClicked(mitk::Point3D pointValue);
  void WheelMoved(QWheelEvent*);
  void WidgetPlanesRotationLinked(bool);
  void WidgetPlanesRotationEnabled(bool);
  void ViewsInitialized();
  void WidgetPlaneModeSlicing(bool);
  void WidgetPlaneModeRotation(bool);
  void WidgetPlaneModeSwivel(bool);
  void WidgetPlaneModeChange(int);
  void WidgetNotifyNewCrossHairMode(int);
  void Moved();

public:

  /** Define RenderWindow (public)*/ 
  QmitkRenderWindow* mitkWidget1;
  QmitkLevelWindowWidget* levelWindowWidget;
  /********************************/

  enum { PLANE_MODE_SLICING = 0, PLANE_MODE_ROTATION, PLANE_MODE_SWIVEL };
  enum { LAYOUT_DEFAULT = 0, LAYOUT_2D_IMAGES_UP, LAYOUT_2D_IMAGES_LEFT,
    LAYOUT_BIG_3D, LAYOUT_WIDGET1, LAYOUT_WIDGET2, LAYOUT_WIDGET3,
    LAYOUT_2X_2D_AND_3D_WIDGET, LAYOUT_ROW_WIDGET_3_AND_4,
    LAYOUT_COLUMN_WIDGET_3_AND_4, LAYOUT_ROW_WIDGET_SMALL3_AND_BIG4 ,
    LAYOUT_SMALL_UPPER_WIDGET2_BIG3_AND4,LAYOUT_2D_AND_3D_LEFT_2D_RIGHT_WIDGET,
    LAYOUT_2D_UP_AND_3D_DOWN};

  enum {
    TRANSVERSAL,
    AXIAL = TRANSVERSAL,
    SAGITTAL,
    CORONAL,
    THREE_D
  };


protected:

  QHBoxLayout* QmitkSingleWidgetLayout;

  int m_Layout;
  int m_PlaneMode;

  mitk::RenderingManager* m_RenderingManager;

  mitk::RenderWindowFrame::Pointer m_RectangleRendering3;
  mitk::RenderWindowFrame::Pointer m_RectangleRendering2;
  mitk::RenderWindowFrame::Pointer m_RectangleRendering1;
  mitk::RenderWindowFrame::Pointer m_RectangleRendering4;

  mitk::ManufacturerLogo::Pointer m_LogoRendering1;
  mitk::ManufacturerLogo::Pointer m_LogoRendering2;
  mitk::ManufacturerLogo::Pointer m_LogoRendering3;
  mitk::ManufacturerLogo::Pointer m_LogoRendering4;

  mitk::GradientBackground::Pointer m_GradientBackground1;
  mitk::GradientBackground::Pointer m_GradientBackground2;
  mitk::GradientBackground::Pointer m_GradientBackground4;
  mitk::GradientBackground::Pointer m_GradientBackground3;
  bool m_GradientBackgroundFlag;
  

  mitk::MouseModeSwitcher::Pointer m_MouseModeSwitcher;
  mitk::CoordinateSupplier::Pointer m_LastLeftClickPositionSupplier;
  mitk::PositionTracker::Pointer m_PositionTracker;
  mitk::SliceNavigationController* m_TimeNavigationController;
  mitk::SlicesRotator::Pointer m_SlicesRotator;
  mitk::SlicesSwiveller::Pointer m_SlicesSwiveller;

  mitk::DataNode::Pointer m_PositionTrackerNode;
  mitk::DataStorage::Pointer m_DataStorage;

  mitk::DataNode::Pointer m_PlaneNode1;
  mitk::DataNode::Pointer m_PlaneNode2;
  mitk::DataNode::Pointer m_PlaneNode3;
  mitk::DataNode::Pointer m_Node;

  QSplitter *m_MainSplit;
  QSplitter *m_LayoutSplit;
  QSplitter *m_SubSplit1;
  QSplitter *m_SubSplit2;
  
  QWidget *mitkWidget1Container;

  struct  
  {
    vtkCornerAnnotation *cornerText;
    vtkTextProperty *textProp;
    vtkRenderer *ren;
  } m_CornerAnnotaions[3];

  bool m_PendingCrosshairPositionEvent;
  bool m_CrosshairNavigationEnabled;
    
};
#endif /*QMITKSIGLEWIDGET_H_*/
