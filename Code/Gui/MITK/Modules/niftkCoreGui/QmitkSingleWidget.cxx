/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#define SMW_INFO MITK_INFO("widget.single")

#include "QmitkSingleWidget.h"

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QGridLayout>
#include <qsplitter.h>
#include <QMotifStyle>
#include <QList>
#include <QMouseEvent>
#include <QTimer>

#include <mitkProperties.h>
#include <mitkGeometry2DDataMapper2D.h>
#include <mitkGlobalInteraction.h>
#include <mitkDisplayInteractor.h>
#include <mitkPointSet.h>
#include <mitkPositionEvent.h>
#include <mitkStateEvent.h>
#include <mitkLine.h>
#include <mitkInteractionConst.h>
#include <mitkDataStorage.h>

#include <mitkNodePredicateBase.h>
#include <mitkNodePredicateDataType.h>

#include <mitkNodePredicateNot.h>
#include <mitkNodePredicateProperty.h>
#include <mitkStatusBar.h>
#include <mitkImage.h>

#include <mitkVtkLayerController.h>


QmitkSingleWidget::QmitkSingleWidget(QWidget* parent, Qt::WindowFlags f, mitk::RenderingManager* renderingManager)
: QWidget(parent, f), 
mitkWidget1(NULL),
levelWindowWidget(NULL),
QmitkSingleWidgetLayout(NULL),
m_Layout(LAYOUT_DEFAULT),
m_PlaneMode(PLANE_MODE_SLICING),
m_RenderingManager(renderingManager),
m_GradientBackgroundFlag(true),
m_TimeNavigationController(NULL),
m_MainSplit(NULL),
m_LayoutSplit(NULL),
mitkWidget1Container(NULL),
m_PendingCrosshairPositionEvent(false),
m_CrosshairNavigationEnabled(false)
{
  /******************************************************
   * Use the global RenderingManager if none was specified
   * ****************************************************/
  if (m_RenderingManager == NULL)
  {
    m_RenderingManager = mitk::RenderingManager::GetInstance();
  }
  m_TimeNavigationController = m_RenderingManager->GetTimeNavigationController();

  /*******************************/
  //Create Widget manually
  /*******************************/

  //create Layouts
  QmitkSingleWidgetLayout = new QHBoxLayout( this ); 
  QmitkSingleWidgetLayout->setContentsMargins(0,0,0,0);

  //Set Layout to widget
  this->setLayout(QmitkSingleWidgetLayout);

//  QmitkNavigationToolBar* toolBar = new QmitkNavigationToolBar();
//  QmitkSingleWidgetLayout->addWidget( toolBar );

  //create main splitter
  m_MainSplit = new QSplitter( this );
  QmitkSingleWidgetLayout->addWidget( m_MainSplit );
  
  //creae Widget Container
  mitkWidget1Container = new QWidget(m_MainSplit);

  mitkWidget1Container->setContentsMargins(0,0,0,0);

  //create Widget Layout
  QHBoxLayout *mitkWidgetLayout1 = new QHBoxLayout(mitkWidget1Container);

  mitkWidgetLayout1->setMargin(0);

  //set Layout to Widget Container  
  mitkWidget1Container->setLayout(mitkWidgetLayout1); 

  //set SizePolicy
  mitkWidget1Container->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Expanding);

  m_MainSplit->addWidget( mitkWidget1Container );

  //  m_RenderingManager->SetGlobalInteraction( mitk::GlobalInteraction::GetInstance() );

  //Create RenderWindows 1
  mitkWidget1 = new QmitkRenderWindow(mitkWidget1Container, "single.widget1", NULL, m_RenderingManager);
  mitkWidget1->setMaximumSize(2000,2000);
  mitkWidget1->SetLayoutIndex( THREE_D );
  mitkWidgetLayout1->addWidget(mitkWidget1); 

  //create SignalSlot Connection
  connect( mitkWidget1, SIGNAL( SignalLayoutDesignChanged(int) ), this, SLOT( OnLayoutDesignChanged(int) ) );
  connect( mitkWidget1, SIGNAL( ResetView() ), this, SLOT( ResetCrosshair() ) );
  connect( mitkWidget1, SIGNAL( ChangeCrosshairRotationMode(int) ), this, SLOT( SetWidgetPlaneMode(int) ) );
  connect( this, SIGNAL(WidgetNotifyNewCrossHairMode(int)), mitkWidget1, SLOT(OnWidgetPlaneModeChanged(int)) );

  //Create Level Window Widget
  levelWindowWidget = new QmitkLevelWindowWidget( m_MainSplit ); //this
  levelWindowWidget->setObjectName(QString::fromUtf8("levelWindowWidget"));
  QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
  sizePolicy.setHorizontalStretch(0);
  sizePolicy.setVerticalStretch(0);
  sizePolicy.setHeightForWidth(levelWindowWidget->sizePolicy().hasHeightForWidth());
  levelWindowWidget->setSizePolicy(sizePolicy);
  levelWindowWidget->setMaximumSize(QSize(50, 2000));
  
  //add LevelWindow Widget to mainSplitter
  m_MainSplit->addWidget( levelWindowWidget );

  //show mainSplitt and add to Layout
  m_MainSplit->show();

  //resize Image.
  this->resize( QSize(364, 477).expandedTo(minimumSizeHint()) );

  //Initialize the widgets.
  this->InitializeWidget();

  //Activate Widget Menu
  this->ActivateMenuWidget( false );
}

float QmitkSingleWidget::GetOpacity() const
{
  return static_cast<float>(m_BitmapOverlay1->GetOpacity());
}

void QmitkSingleWidget::SetOpacity(const float& value)
{
  m_BitmapOverlay1->SetOpacity(value);
}


void QmitkSingleWidget::SetImageNode(const mitk::DataNode* node)
{
  m_BitmapOverlay1->SetNode(node);
}


void QmitkSingleWidget::SetTransformNode(const mitk::DataNode* node)
{

}

void QmitkSingleWidget::InitializeWidget()
{
  m_PositionTracker = NULL;

  // transfer colors in WorldGeometry-Nodes of the associated Renderer
  QColor qcolor;
  //float color[3] = {1.0f,1.0f,1.0f};
  mitk::DataNode::Pointer planeNode;
  mitk::IntProperty::Pointer  layer;

  // of widget 1
  planeNode = mitk::BaseRenderer::GetInstance(mitkWidget1->GetRenderWindow())->GetCurrentWorldGeometry2DNode();
  planeNode->SetColor(1.0,0.0,0.0);
  layer = mitk::IntProperty::New(1000);
  planeNode->SetProperty("layer",layer);

  mitk::BaseRenderer::GetInstance(mitkWidget1->GetRenderWindow())->SetMapperID(mitk::BaseRenderer::Standard3D);
  // Set plane mode (slicing/rotation behavior) to slicing (default)
  m_PlaneMode = PLANE_MODE_SLICING;

  // Set default view directions for SNCs
  mitkWidget1->GetSliceNavigationController()->SetDefaultViewDirection(
    mitk::SliceNavigationController::Original );

  /*************************************************/
  //Write Layout Names into the viewers -- hardCoded

  //Info for later: 
  //int view = this->GetRenderWindow1()->GetSliceNavigationController()->GetDefaultViewDirection();
  //QString layoutName;
  //if( view == mitk::SliceNavigationController::Axial )
  //  layoutName = "Axial";
  //else if( view == mitk::SliceNavigationController::Sagittal )
  //  layoutName = "Sagittal";
  //else if( view == mitk::SliceNavigationController::Frontal )
  //  layoutName = "Coronal";
  //else if( view == mitk::SliceNavigationController::Original )
  //  layoutName = "Original";
  //if( view >= 0 && view < 4 )
  //  //write LayoutName --> Viewer 3D shoudn't write the layoutName.

  //Render Window 1 == axial
  m_CornerAnnotaions[0].cornerText = vtkCornerAnnotation::New();
  m_CornerAnnotaions[0].cornerText->SetText(0, "Single View");
  m_CornerAnnotaions[0].cornerText->SetMaximumFontSize(12);
  m_CornerAnnotaions[0].textProp = vtkTextProperty::New();
  m_CornerAnnotaions[0].textProp->SetColor( 1.0, 0.0, 0.0 );
  m_CornerAnnotaions[0].cornerText->SetTextProperty( m_CornerAnnotaions[0].textProp );
  m_CornerAnnotaions[0].ren = vtkRenderer::New();
  m_CornerAnnotaions[0].ren->AddActor(m_CornerAnnotaions[0].cornerText);
  m_CornerAnnotaions[0].ren->InteractiveOff();
  mitk::VtkLayerController::GetInstance(this->GetRenderWindow1()->GetRenderWindow())->InsertForegroundRenderer(m_CornerAnnotaions[0].ren,true);
  


  // create a slice rotator
  // m_SlicesRotator = mitk::SlicesRotator::New();
  // @TODO next line causes sure memory leak
  // rotator will be created nonetheless (will be switched on and off)
  m_SlicesRotator = mitk::SlicesRotator::New("slices-rotator");
  m_SlicesRotator->AddSliceController(
    mitkWidget1->GetSliceNavigationController() );

  // create a slice swiveller (using the same state-machine as SlicesRotator)
  m_SlicesSwiveller = mitk::SlicesSwiveller::New("slices-rotator");
  m_SlicesSwiveller->AddSliceController(
    mitkWidget1->GetSliceNavigationController() );

  //connect to the "time navigation controller": send time via sliceNavigationControllers
  m_TimeNavigationController->ConnectGeometryTimeEvent(
    mitkWidget1->GetSliceNavigationController() , false);

  //reverse connection between sliceNavigationControllers and m_TimeNavigationController
  mitkWidget1->GetSliceNavigationController()
    ->ConnectGeometryTimeEvent(m_TimeNavigationController, false);

  m_MouseModeSwitcher = mitk::MouseModeSwitcher::New();

  m_LastLeftClickPositionSupplier =
    mitk::CoordinateSupplier::New("navigation", NULL);
  mitk::GlobalInteraction::GetInstance()->AddListener(
    m_LastLeftClickPositionSupplier
    );
  // setup gradient background
  m_GradientBackground1 = mitk::GradientBackground::New();
  m_GradientBackground1->SetRenderWindow(
    mitkWidget1->GetRenderWindow() );
  m_GradientBackground1->SetGradientColors(0.1,0.1,0.1,0.5,0.5,0.5);
  m_GradientBackground1->Enable();

  // setup the department logo rendering
  //m_LogoRendering1 = mitk::ManufacturerLogo::New();
  m_LogoRendering1 = CMICLogo::New();
  m_LogoRendering1->SetRenderWindow(
    mitkWidget1->GetRenderWindow() );
  m_LogoRendering1->Disable();

  m_BitmapOverlay1 = QmitkBitmapOverlay::New();
  m_BitmapOverlay1->SetRenderWindow(
    mitkWidget1->GetRenderWindow() );

  m_RectangleRendering1 = mitk::RenderWindowFrame::New();
  m_RectangleRendering1->SetRenderWindow(
    mitkWidget1->GetRenderWindow() );
  m_RectangleRendering1->Enable(1.0,0.0,0.0);
}
 
void QmitkSingleWidget::EnableDepartmentLogo()
{
   m_LogoRendering1->Enable();
}
void QmitkSingleWidget::DisableDepartmentLogo()
{
   m_LogoRendering1->Disable();
}

QmitkSingleWidget::~QmitkSingleWidget()
{
  DisablePositionTracking();
  DisableNavigationControllerEventListening();

  m_TimeNavigationController->Disconnect(mitkWidget1->GetSliceNavigationController());

  mitk::VtkLayerController::GetInstance(this->GetRenderWindow1()->GetRenderWindow())->RemoveRenderer( m_CornerAnnotaions[0].ren );

  //Delete CornerAnnotation
  m_CornerAnnotaions[0].cornerText->Delete();
  m_CornerAnnotaions[0].textProp->Delete();
  m_CornerAnnotaions[0].ren->Delete();

}

void QmitkSingleWidget::RemoveCameraFromDataStorage()
{
  if (m_PlaneNode1.IsNotNull() && m_PlaneNode2.IsNotNull() && m_PlaneNode3.IsNotNull() && m_Node.IsNotNull())
  {
    if(m_DataStorage.IsNotNull())
    {
      m_DataStorage->Remove(m_PlaneNode1);
      m_DataStorage->Remove(m_PlaneNode2);
      m_DataStorage->Remove(m_PlaneNode3);
      m_DataStorage->Remove(m_Node);
    }
  }
}

void QmitkSingleWidget::AddCameraToDataStorage()
{
  vtkCamera * Camera;
  Camera = mitk::BaseRenderer::GetInstance(mitkWidget1->GetRenderWindow())->GetVtkRenderer()->GetActiveCamera();
  
  if (Camera != NULL  )
  {
    if (m_DataStorage.IsNotNull())
    {
      m_DataStorage->Add(m_Node);
      m_DataStorage->Add(m_PlaneNode1, m_Node);
      m_DataStorage->Add(m_PlaneNode2, m_Node);
      m_DataStorage->Add(m_PlaneNode3, m_Node);
      static_cast<mitk::Geometry2DDataMapper2D*>(m_PlaneNode1->GetMapper(mitk::BaseRenderer::Standard2D))->SetDatastorageAndGeometryBaseNode(m_DataStorage, m_Node);
      static_cast<mitk::Geometry2DDataMapper2D*>(m_PlaneNode2->GetMapper(mitk::BaseRenderer::Standard2D))->SetDatastorageAndGeometryBaseNode(m_DataStorage, m_Node);
      static_cast<mitk::Geometry2DDataMapper2D*>(m_PlaneNode3->GetMapper(mitk::BaseRenderer::Standard2D))->SetDatastorageAndGeometryBaseNode(m_DataStorage, m_Node);
    }
  }
}


void QmitkSingleWidget::changeLayoutToDefault()
{
  SMW_INFO << "changing layout to default... " << std::endl;
  this->changeLayoutToBig3D();
}

void QmitkSingleWidget::changeLayoutToBig3D()
{
  SMW_INFO << "changing layout to big 3D ..." << std::endl;

  //Hide all Menu Widgets
  this->HideAllWidgetToolbars();

  delete QmitkSingleWidgetLayout ;

  //create Main Layout
  QmitkSingleWidgetLayout =  new QHBoxLayout( this );

  //create main splitter
  m_MainSplit = new QSplitter( this );
  QmitkSingleWidgetLayout->addWidget( m_MainSplit );

  //add widget Splitter to main Splitter
  m_MainSplit->addWidget( mitkWidget1Container );

  //add LevelWindow Widget to mainSplitter
  m_MainSplit->addWidget( levelWindowWidget );

  //show mainSplitt and add to Layout
  m_MainSplit->show();

  //show/hide Widgets
  if ( mitkWidget1->isHidden() ) mitkWidget1->show();

  m_Layout = LAYOUT_BIG_3D;

  //update Layout Design List
  mitkWidget1->LayoutDesignListChanged( LAYOUT_BIG_3D );

  //update Alle Widgets
  this->UpdateAllWidgets();
}


void QmitkSingleWidget::SetDataStorage( mitk::DataStorage* ds )
{
  mitk::BaseRenderer::GetInstance(mitkWidget1->GetRenderWindow())->SetDataStorage(ds);
  m_DataStorage = ds;
  //see if we can find a suitable data node
  mitk::DataNode::Pointer dataNode;
  if ( ! m_DataStorage.IsNull())
  {
     m_BitmapOverlay1->SetDataStorage (m_DataStorage);
  }
  m_BitmapOverlay1->Enable();
}


void QmitkSingleWidget::Fit()
{
  vtkRenderer * vtkrenderer;
  mitk::BaseRenderer::GetInstance(mitkWidget1->GetRenderWindow())->GetDisplayGeometry()->Fit();

  int w = vtkObject::GetGlobalWarningDisplay();
  vtkObject::GlobalWarningDisplayOff();

  vtkrenderer = mitk::BaseRenderer::GetInstance(mitkWidget1->GetRenderWindow())->GetVtkRenderer();
  if ( vtkrenderer!= NULL ) 
    vtkrenderer->ResetCamera();

  vtkObject::SetGlobalWarningDisplay(w);
}


void QmitkSingleWidget::InitPositionTracking()
{
  //PoinSetNode for MouseOrientation
  m_PositionTrackerNode = mitk::DataNode::New();
  m_PositionTrackerNode->SetProperty("name", mitk::StringProperty::New("Mouse Position"));
  m_PositionTrackerNode->SetData( mitk::PointSet::New() );
  m_PositionTrackerNode->SetColor(1.0,0.33,0.0);
  m_PositionTrackerNode->SetProperty("layer", mitk::IntProperty::New(1001));
  m_PositionTrackerNode->SetVisibility(true);
  m_PositionTrackerNode->SetProperty("inputdevice", mitk::BoolProperty::New(true) );
  m_PositionTrackerNode->SetProperty("BaseRendererMapperID", mitk::IntProperty::New(0) );//point position 2D mouse
  m_PositionTrackerNode->SetProperty("baserenderer", mitk::StringProperty::New("N/A"));
}


void QmitkSingleWidget::AddDisplayPlaneSubTree()
{
  // add the displayed planes of the multiwidget to a node to which the subtree 
  // @a planesSubTree points ...
  mitk::Geometry2DDataMapper2D::Pointer mapper;
  // ... of widget 1
  m_PlaneNode1 = (mitk::BaseRenderer::GetInstance(mitkWidget1->GetRenderWindow()))->GetCurrentWorldGeometry2DNode();
  m_PlaneNode1->SetProperty("visible", mitk::BoolProperty::New(true));
  m_PlaneNode1->SetProperty("name", mitk::StringProperty::New("widget1Plane"));
  m_PlaneNode1->SetProperty("includeInBoundingBox", mitk::BoolProperty::New(false));
  m_PlaneNode1->SetProperty("helper object", mitk::BoolProperty::New(true));
  mapper = mitk::Geometry2DDataMapper2D::New();
  m_PlaneNode1->SetMapper(mitk::BaseRenderer::Standard2D, mapper);

  m_Node = mitk::DataNode::New();
  m_Node->SetProperty("name", mitk::StringProperty::New("Widgets"));
  m_Node->SetProperty("helper object", mitk::BoolProperty::New(true));
}


mitk::SliceNavigationController* QmitkSingleWidget::GetTimeNavigationController()
{
  return m_TimeNavigationController;
}


void QmitkSingleWidget::EnableStandardLevelWindow()
{
  levelWindowWidget->disconnect(this);
  levelWindowWidget->SetDataStorage(mitk::BaseRenderer::GetInstance(mitkWidget1->GetRenderWindow())->GetDataStorage());
  levelWindowWidget->show();
}


void QmitkSingleWidget::DisableStandardLevelWindow()
{
  levelWindowWidget->disconnect(this);
  levelWindowWidget->hide();
}


// CAUTION: Legacy code for enabling Qt-signal-controlled view initialization.
// Use RenderingManager::InitializeViews() instead.
bool QmitkSingleWidget::InitializeStandardViews( const mitk::Geometry3D * geometry )
{
  return m_RenderingManager->InitializeViews( geometry );
}


void QmitkSingleWidget::RequestUpdate()
{
  m_RenderingManager->RequestUpdate(mitkWidget1->GetRenderWindow());
}


void QmitkSingleWidget::ForceImmediateUpdate()
{
  m_RenderingManager->ForceImmediateUpdate(mitkWidget1->GetRenderWindow());
}


void QmitkSingleWidget::wheelEvent( QWheelEvent * e )
{
  emit WheelMoved( e );
}

void QmitkSingleWidget::mousePressEvent(QMouseEvent * e)
{
   if (e->button() == Qt::LeftButton) {
       mitk::Point3D pointValue = this->GetLastLeftClickPosition();
       emit LeftMouseClicked(pointValue); 
   }
}

void QmitkSingleWidget::moveEvent( QMoveEvent* e )
{
  QWidget::moveEvent( e );
  
  // it is necessary to readjust the position of the overlays as the SingleWidget has moved
  // unfortunately it's not done by QmitkRenderWindow::moveEvent -> must be done here
  emit Moved();
}

void QmitkSingleWidget::leaveEvent ( QEvent * /*e*/  )
{
  //set cursor back to initial state
  m_SlicesRotator->ResetMouseCursor();
}

QmitkRenderWindow* QmitkSingleWidget::GetRenderWindow1() const
{
  return mitkWidget1;
}


const mitk::Point3D& QmitkSingleWidget::GetLastLeftClickPosition() const
{
  return m_LastLeftClickPositionSupplier->GetCurrentPoint();
}


const mitk::Point3D QmitkSingleWidget::GetCrossPosition() const
{
  return m_LastLeftClickPositionSupplier->GetCurrentPoint();
}


void QmitkSingleWidget::EnablePositionTracking()
{
  if (!m_PositionTracker)
  {
    m_PositionTracker = mitk::PositionTracker::New("PositionTracker", NULL);
  }
  mitk::GlobalInteraction* globalInteraction = mitk::GlobalInteraction::GetInstance();
  if (globalInteraction)
  {
    if(m_DataStorage.IsNotNull())
      m_DataStorage->Add(m_PositionTrackerNode);
    globalInteraction->AddListener(m_PositionTracker);
  }
}


void QmitkSingleWidget::DisablePositionTracking()
{
  mitk::GlobalInteraction* globalInteraction =
    mitk::GlobalInteraction::GetInstance();

  if(globalInteraction)
  {
    if (m_DataStorage.IsNotNull())
      m_DataStorage->Remove(m_PositionTrackerNode);
    globalInteraction->RemoveListener(m_PositionTracker);
  }
}


void QmitkSingleWidget::EnsureDisplayContainsPoint(
  mitk::DisplayGeometry* displayGeometry, const mitk::Point3D& p)
{
  mitk::Point2D pointOnPlane;
  displayGeometry->Map( p, pointOnPlane );

  // point minus origin < width or height ==> outside ?
  mitk::Vector2D pointOnRenderWindow_MM;
  pointOnRenderWindow_MM = pointOnPlane.GetVectorFromOrigin()
    - displayGeometry->GetOriginInMM();

  mitk::Vector2D sizeOfDisplay( displayGeometry->GetSizeInMM() );

  if (   sizeOfDisplay[0] < pointOnRenderWindow_MM[0]
  ||                0 > pointOnRenderWindow_MM[0]
  || sizeOfDisplay[1] < pointOnRenderWindow_MM[1]
  ||                0 > pointOnRenderWindow_MM[1] )
  {
    // point is not visible -> move geometry
    mitk::Vector2D offset( (pointOnRenderWindow_MM - sizeOfDisplay / 2.0)
      / displayGeometry->GetScaleFactorMMPerDisplayUnit() );

    displayGeometry->MoveBy( offset );
  }
}


void QmitkSingleWidget::MoveCrossToPosition(const mitk::Point3D& newPosition)
{
  // create a PositionEvent with the given position and
  // tell the slice navigation controllers to move there

  mitk::Point2D p2d;
  mitk::PositionEvent event( mitk::BaseRenderer::GetInstance(mitkWidget1->GetRenderWindow()), 0, 0, 0,
    mitk::Key_unknown, p2d, newPosition );
  mitk::StateEvent stateEvent(mitk::EIDLEFTMOUSEBTN, &event);
  mitk::StateEvent stateEvent2(mitk::EIDLEFTMOUSERELEASE, &event);

  switch ( m_PlaneMode )
  {
  default:
  case PLANE_MODE_SLICING:
    mitkWidget1->GetSliceNavigationController()->HandleEvent( &stateEvent );

    // just in case SNCs will develop something that depends on the mouse
    // button being released again
    mitkWidget1->GetSliceNavigationController()->HandleEvent( &stateEvent2 );
    break;

  case PLANE_MODE_ROTATION:
    m_SlicesRotator->HandleEvent( &stateEvent );

    // just in case SNCs will develop something that depends on the mouse
    // button being released again
    m_SlicesRotator->HandleEvent( &stateEvent2 );
    break;

  case PLANE_MODE_SWIVEL:
    m_SlicesSwiveller->HandleEvent( &stateEvent );

    // just in case SNCs will develop something that depends on the mouse
    // button being released again
    m_SlicesSwiveller->HandleEvent( &stateEvent2 );
    break;
  }

  // determine if cross is now out of display
  // if so, move the display window
  EnsureDisplayContainsPoint( mitk::BaseRenderer::GetInstance(mitkWidget1->GetRenderWindow())
    ->GetDisplayGeometry(), newPosition );

  // update displays
  m_RenderingManager->RequestUpdateAll();
}

void QmitkSingleWidget::HandleCrosshairPositionEvent()
{
  if(!m_PendingCrosshairPositionEvent)
  {
    m_PendingCrosshairPositionEvent=true;
    QTimer::singleShot(0,this,SLOT( HandleCrosshairPositionEventDelayed() ) );
  }
}

void QmitkSingleWidget::HandleCrosshairPositionEventDelayed()
{
  std::string statusText;
  std::stringstream stream;
  stream << "Cross hair position event";

  statusText = stream.str();
  mitk::StatusBar::GetInstance()->DisplayGreyValueText(statusText.c_str());


}

void QmitkSingleWidget::EnableNavigationControllerEventListening()
{
  // Let NavigationControllers listen to GlobalInteraction
  mitk::GlobalInteraction *gi = mitk::GlobalInteraction::GetInstance();

  // Listen for SliceNavigationController
  mitkWidget1->GetSliceNavigationController()->crosshairPositionEvent.AddListener( mitk::MessageDelegate<QmitkSingleWidget>( this, &QmitkSingleWidget::HandleCrosshairPositionEvent ) );

  switch ( m_PlaneMode )
  {
  default:
  case PLANE_MODE_SLICING:
    gi->AddListener( mitkWidget1->GetSliceNavigationController() );
    break;

  case PLANE_MODE_ROTATION:
    gi->AddListener( m_SlicesRotator );
    break;

  case PLANE_MODE_SWIVEL:
    gi->AddListener( m_SlicesSwiveller );
    break;
  }

  gi->AddListener( m_TimeNavigationController );
  m_CrosshairNavigationEnabled = true;
}

void QmitkSingleWidget::DisableNavigationControllerEventListening()
{
  // Do not let NavigationControllers listen to GlobalInteraction
  mitk::GlobalInteraction *gi = mitk::GlobalInteraction::GetInstance();

  switch ( m_PlaneMode )
  {
  default:
  case PLANE_MODE_SLICING:
    gi->RemoveListener( mitkWidget1->GetSliceNavigationController() );
    break;

  case PLANE_MODE_ROTATION:
    m_SlicesRotator->ResetMouseCursor();
    gi->RemoveListener( m_SlicesRotator );
    break;

  case PLANE_MODE_SWIVEL:
    m_SlicesSwiveller->ResetMouseCursor();
    gi->RemoveListener( m_SlicesSwiveller );
    break;
  }

  gi->RemoveListener( m_TimeNavigationController );
  m_CrosshairNavigationEnabled = false;
}


int QmitkSingleWidget::GetLayout() const
{
  return m_Layout;
}

bool QmitkSingleWidget::GetGradientBackgroundFlag() const
{
  return m_GradientBackgroundFlag;
}

void QmitkSingleWidget::EnableGradientBackground()
{
  // gradient background is by default only in widget 4, otherwise
  // interferences between 2D rendering and VTK rendering may occur.
  m_GradientBackground1->Enable();
  m_GradientBackgroundFlag = true;
}


void QmitkSingleWidget::DisableGradientBackground()
{
  m_GradientBackground1->Disable();
  m_GradientBackgroundFlag = false;
}


bool QmitkSingleWidget::IsCrosshairNavigationEnabled() const
{
  return m_CrosshairNavigationEnabled;
}


mitk::SlicesRotator * QmitkSingleWidget::GetSlicesRotator() const
{
  return m_SlicesRotator;
}


mitk::SlicesSwiveller * QmitkSingleWidget::GetSlicesSwiveller() const
{
  return m_SlicesSwiveller;
}


void QmitkSingleWidget::SetWidgetPlaneVisibility(const char* widgetName, bool visible, mitk::BaseRenderer *renderer)
{
  if (m_DataStorage.IsNotNull())
  {
    mitk::DataNode* n = m_DataStorage->GetNamedNode(widgetName);
    if (n != NULL)
      n->SetVisibility(visible, renderer);
  }
}


void QmitkSingleWidget::SetWidgetPlanesVisibility(bool visible, mitk::BaseRenderer *renderer)
{
  SetWidgetPlaneVisibility("widget1Plane", visible, renderer);
  SetWidgetPlaneVisibility("widget2Plane", visible, renderer);
  SetWidgetPlaneVisibility("widget3Plane", visible, renderer);
  m_RenderingManager->RequestUpdateAll();
}


void QmitkSingleWidget::SetWidgetPlanesLocked(bool locked)
{
  //do your job and lock or unlock slices.
  GetRenderWindow1()->GetSliceNavigationController()->SetSliceLocked(locked);
}


void QmitkSingleWidget::SetWidgetPlanesRotationLocked(bool locked)
{
  //do your job and lock or unlock slices.
  GetRenderWindow1()->GetSliceNavigationController()->SetSliceRotationLocked(locked);
}


void QmitkSingleWidget::SetWidgetPlanesRotationLinked( bool link )
{
  m_SlicesRotator->SetLinkPlanes( link );
  m_SlicesSwiveller->SetLinkPlanes( link );
  emit WidgetPlanesRotationLinked( link );
}


void QmitkSingleWidget::SetWidgetPlaneMode( int userMode )
{
  MITK_DEBUG << "Changing crosshair mode to " << userMode;

  // first of all reset left mouse button interaction to default if PACS interaction style is active
  m_MouseModeSwitcher->SelectMouseMode( mitk::MouseModeSwitcher::MousePointer );

  emit WidgetNotifyNewCrossHairMode( userMode );
  
  int mode = m_PlaneMode;
  bool link = false;
  
  // Convert user interface mode to actual mode
  {
    switch(userMode)
    {
      case 0:
        mode = PLANE_MODE_SLICING;
        link = false;
        break;
      
      case 1:
        mode = PLANE_MODE_ROTATION;
        link = false;
        break;
     
      case 2:
        mode = PLANE_MODE_ROTATION;
        link = true;
        break;
     
      case 3:
        mode = PLANE_MODE_SWIVEL;
        link = false;
        break;
    }
  }

  // Slice rotation linked
  m_SlicesRotator->SetLinkPlanes( link );
  m_SlicesSwiveller->SetLinkPlanes( link );

  // Do nothing if mode didn't change
  if ( m_PlaneMode == mode )
  {
    return;
  }

  mitk::GlobalInteraction *gi = mitk::GlobalInteraction::GetInstance();

  // Remove listeners of previous mode
  switch ( m_PlaneMode )
  {
  default:
  case PLANE_MODE_SLICING:
    // Notify MainTemplate GUI that this mode has been deselected
    emit WidgetPlaneModeSlicing( false );

    gi->RemoveListener( mitkWidget1->GetSliceNavigationController() );
    break;

  case PLANE_MODE_ROTATION:
    // Notify MainTemplate GUI that this mode has been deselected
    emit WidgetPlaneModeRotation( false );

    m_SlicesRotator->ResetMouseCursor();
    gi->RemoveListener( m_SlicesRotator );
    break;

  case PLANE_MODE_SWIVEL:
    // Notify MainTemplate GUI that this mode has been deselected
    emit WidgetPlaneModeSwivel( false );

    m_SlicesSwiveller->ResetMouseCursor();
    gi->RemoveListener( m_SlicesSwiveller );
    break;
  }

  // Set new mode and add corresponding listener to GlobalInteraction
  m_PlaneMode = mode;
  switch ( m_PlaneMode )
  {
  default:
  case PLANE_MODE_SLICING:
    // Notify MainTemplate GUI that this mode has been selected
    emit WidgetPlaneModeSlicing( true );

    // Add listeners
    gi->AddListener( mitkWidget1->GetSliceNavigationController() );

    m_RenderingManager->InitializeViews();
    break;

  case PLANE_MODE_ROTATION:
    // Notify MainTemplate GUI that this mode has been selected
    emit WidgetPlaneModeRotation( true );

    // Add listener
    gi->AddListener( m_SlicesRotator );
    break;

  case PLANE_MODE_SWIVEL:
    // Notify MainTemplate GUI that this mode has been selected
    emit WidgetPlaneModeSwivel( true );

    // Add listener
    gi->AddListener( m_SlicesSwiveller );
    break;
  }
  // Notify MainTemplate GUI that mode has changed
  emit WidgetPlaneModeChange(m_PlaneMode);
}


void QmitkSingleWidget::SetGradientBackgroundColors( const mitk::Color & upper, const mitk::Color & lower )
{
  m_GradientBackground1->SetGradientColors(upper[0], upper[1], upper[2], lower[0], lower[1], lower[2]);
  m_GradientBackgroundFlag = true;
}


void QmitkSingleWidget::SetDepartmentLogoPath( const char * path )
{
  m_LogoRendering1->SetLogoSource(path);
}


void QmitkSingleWidget::SetWidgetPlaneModeToSlicing( bool activate )
{
  if ( activate )
  {
    this->SetWidgetPlaneMode( PLANE_MODE_SLICING );
  }
}


void QmitkSingleWidget::SetWidgetPlaneModeToRotation( bool activate )
{
  if ( activate )
  {
    this->SetWidgetPlaneMode( PLANE_MODE_ROTATION );
  }
}


void QmitkSingleWidget::SetWidgetPlaneModeToSwivel( bool activate )
{
  if ( activate )
  {
    this->SetWidgetPlaneMode( PLANE_MODE_SWIVEL );
  }
}

void QmitkSingleWidget::OnLayoutDesignChanged( int layoutDesignIndex )
{
  switch( layoutDesignIndex )
  {
  case LAYOUT_DEFAULT:
    {
      this->changeLayoutToDefault();
      break;
    }
  case LAYOUT_BIG_3D:
    {
      this->changeLayoutToBig3D();
      break;
    }

  };


}

void QmitkSingleWidget::UpdateAllWidgets()
{
  mitkWidget1->resize( mitkWidget1Container->frameSize().width()-1, mitkWidget1Container->frameSize().height() );
  mitkWidget1->resize( mitkWidget1Container->frameSize().width(), mitkWidget1Container->frameSize().height() );




}


void QmitkSingleWidget::HideAllWidgetToolbars()
{
  mitkWidget1->HideRenderWindowMenu();
}

void QmitkSingleWidget::ActivateMenuWidget( bool state )
{
  mitkWidget1->ActivateMenuWidget( state  );
}

bool QmitkSingleWidget::IsMenuWidgetEnabled() const
{
  return mitkWidget1->GetActivateMenuWidgetFlag();
}
 
void QmitkSingleWidget::ResetCrosshair()
{
  if (m_DataStorage.IsNotNull())
  {
    mitk::NodePredicateNot::Pointer pred
    = mitk::NodePredicateNot::New(mitk::NodePredicateProperty::New("includeInBoundingBox"
    , mitk::BoolProperty::New(false)));

    mitk::NodePredicateNot::Pointer pred2
    = mitk::NodePredicateNot::New(mitk::NodePredicateProperty::New("includeInBoundingBox"
    , mitk::BoolProperty::New(true)));

    mitk::DataStorage::SetOfObjects::ConstPointer rs = m_DataStorage->GetSubset(pred);
    mitk::DataStorage::SetOfObjects::ConstPointer rs2 = m_DataStorage->GetSubset(pred2);
    // calculate bounding geometry of these nodes
    mitk::TimeSlicedGeometry::Pointer bounds = m_DataStorage->ComputeBoundingGeometry3D(rs, "visible");
    
    m_RenderingManager->InitializeViews(bounds);
    //m_RenderingManager->InitializeViews( m_DataStorage->ComputeVisibleBoundingGeometry3D() );
    // reset interactor to normal slicing
    this->SetWidgetPlaneMode(PLANE_MODE_SLICING);
  }
}

void QmitkSingleWidget::EnableColoredRectangles()
{
  m_RectangleRendering1->Enable(1.0, 0.0, 0.0);
}

void QmitkSingleWidget::DisableColoredRectangles()
{
  m_RectangleRendering1->Disable();
}

bool QmitkSingleWidget::IsColoredRectanglesEnabled() const
{
  return m_RectangleRendering1->IsEnabled();
}

mitk::MouseModeSwitcher* QmitkSingleWidget::GetMouseModeSwitcher()
{
  return m_MouseModeSwitcher;
}

void QmitkSingleWidget::MouseModeSelected( mitk::MouseModeSwitcher::MouseMode mouseMode )
{
  if ( mouseMode == 0 )
  {
    this->EnableNavigationControllerEventListening();
  }
  else
  {
    this->DisableNavigationControllerEventListening();
  }
}

mitk::DataNode::Pointer QmitkSingleWidget::GetWidgetPlane1()
{
  return m_PlaneNode1;
}

mitk::DataNode::Pointer QmitkSingleWidget::GetWidgetPlane2()
{
  return m_PlaneNode2;
}

mitk::DataNode::Pointer QmitkSingleWidget::GetWidgetPlane3()
{
  return m_PlaneNode3;
}

mitk::DataNode::Pointer QmitkSingleWidget::GetWidgetPlane(int id)
{
  switch(id)
  {
    case 1: return m_PlaneNode1;
    break;
    case 2: return m_PlaneNode2;
    break;
    case 3: return m_PlaneNode3;
    break;
    default: return NULL;
  }
}

