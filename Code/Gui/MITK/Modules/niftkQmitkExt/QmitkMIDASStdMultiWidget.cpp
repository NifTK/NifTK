/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $LastChangedDate: 2011-12-16 09:12:58 +0000 (Fri, 16 Dec 2011) $
 Revision          : $Revision: 8039 $
 Last modified by  : $Author: mjc $

 Original author   : $Author: mjc $

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "QmitkMIDASStdMultiWidget.h"
#include "QmitkRenderWindow.h"
#include "vtkRenderWindow.h"
#include <QStackedLayout>
#include <QGridLayout>
#include <QFrame>

QmitkMIDASStdMultiWidget::QmitkMIDASStdMultiWidget(
    mitk::RenderingManager* renderingManager,
    mitk::DataStorage* dataStorage,
    QWidget* parent,
    Qt::WindowFlags f
    )
: QmitkStdMultiWidget(parent, f, renderingManager)
, m_GridLayout(NULL)
, m_AxialSliceTag(0)
, m_SagittalSliceTag(0)
, m_CoronalSliceTag(0)
, m_IsSelected(false)
, m_IsEnabled(false)
, m_Display2DCursorsLocally(true)
, m_Display2DCursorsGlobally(false)
, m_View(MIDAS_VIEW_ORTHO)
, m_MagnificationFactor(0)
{
  // The spec in the header file says these must be non-null.
  assert(renderingManager);
  assert(dataStorage);

  // As soon as the base class is created, de-register the windows with the provided RenderingManager.
  // This stops a renderer calling m_RenderingManager->RequestUpdateAll when widget is not visible.
  // Then we re-register them, as soon as we are given a valid geometry to work with.
  m_RenderingManager = renderingManager;
  m_RenderingManager->RemoveRenderWindow(this->mitkWidget1->GetVtkRenderWindow());
  m_RenderingManager->RemoveRenderWindow(this->mitkWidget2->GetVtkRenderWindow());
  m_RenderingManager->RemoveRenderWindow(this->mitkWidget3->GetVtkRenderWindow());
  m_RenderingManager->RemoveRenderWindow(this->mitkWidget4->GetVtkRenderWindow());

  // Store this immediately, calling base class method.
  this->SetDataStorage(dataStorage);

  // See also SetEnabled(bool) to see things that are dynamically on/off
  this->HideAllWidgetToolbars();
  this->DisableStandardLevelWindow();
  this->DisableDepartmentLogo();
  this->ActivateMenuWidget(false);
  this->SetBackgroundColor(QColor(0, 0, 0));

  // 3D planes should only be visible in this specific widget, not globally, so we create them, then make them globally invisible.
  this->AddDisplayPlaneSubTree();
  this->SetDisplay2DCursorsGlobally(false);
  this->SetDisplay2DCursorsLocally(false);
  this->SetWidgetPlanesLocked(false);
  this->SetWidgetPlanesRotationLocked(true);

  // Need each widget to react to Qt drag/drop events.
  this->mitkWidget1->setAcceptDrops(true);
  this->mitkWidget2->setAcceptDrops(true);
  this->mitkWidget3->setAcceptDrops(true);
  this->mitkWidget4->setAcceptDrops(true);

  // Turn off all interactors in base class, until we want them to be enabled.
  this->GetMouseModeSwitcher()->SetInteractionScheme(mitk::MouseModeSwitcher::OFF);

  // Set these off, as it wont matter until there is an image dropped, with a specific layout and orientation.
  this->m_CornerAnnotaions[0].cornerText->SetText(0, "");
  this->m_CornerAnnotaions[1].cornerText->SetText(0, "");
  this->m_CornerAnnotaions[2].cornerText->SetText(0, "");

  // Set default layout. Regardless of what you put in parameter1, eg. MIDAS_VIEW_ORTHO
  // effectively by default all the widgets are enabled in the base class. I have not
  // hidden them, because if you do hide them, you get lots of InvalidDrawable errors
  // on a Mac, which while not fatal, are pretty ugly.  So, by default each widget
  // comes up looking like an ortho-view.  This is no bad thing, as it will remind
  // MIDAS people that this application has more functionality than MIDAS.
  this->SetMIDASView(MIDAS_VIEW_ORTHO, true);

  // Default to unselected, so borders are off.
  this->SetSelected(false);

  // Need each widget to signal when something is dropped, so connect signals to OnNodesDropped.
  connect(this->mitkWidget1, SIGNAL(NodesDropped(QmitkRenderWindow*, std::vector<mitk::DataNode*>)), this, SLOT(OnNodesDropped(QmitkRenderWindow*, std::vector<mitk::DataNode*>)));
  connect(this->mitkWidget2, SIGNAL(NodesDropped(QmitkRenderWindow*, std::vector<mitk::DataNode*>)), this, SLOT(OnNodesDropped(QmitkRenderWindow*, std::vector<mitk::DataNode*>)));
  connect(this->mitkWidget3, SIGNAL(NodesDropped(QmitkRenderWindow*, std::vector<mitk::DataNode*>)), this, SLOT(OnNodesDropped(QmitkRenderWindow*, std::vector<mitk::DataNode*>)));
  connect(this->mitkWidget4, SIGNAL(NodesDropped(QmitkRenderWindow*, std::vector<mitk::DataNode*>)), this, SLOT(OnNodesDropped(QmitkRenderWindow*, std::vector<mitk::DataNode*>)));

  // Register to listen to SliceNavigators, slice changed events.
  itk::ReceptorMemberCommand<QmitkMIDASStdMultiWidget>::Pointer onAxialSliceChangedCommand =
    itk::ReceptorMemberCommand<QmitkMIDASStdMultiWidget>::New();
  onAxialSliceChangedCommand->SetCallbackFunction( this, &QmitkMIDASStdMultiWidget::OnAxialSliceChanged );
  m_AxialSliceTag = mitkWidget1->GetSliceNavigationController()->AddObserver(mitk::SliceNavigationController::GeometrySliceEvent(NULL, 0), onAxialSliceChangedCommand);

  itk::ReceptorMemberCommand<QmitkMIDASStdMultiWidget>::Pointer onSagittalSliceChangedCommand =
    itk::ReceptorMemberCommand<QmitkMIDASStdMultiWidget>::New();
  onSagittalSliceChangedCommand->SetCallbackFunction( this, &QmitkMIDASStdMultiWidget::OnSagittalSliceChanged );
  m_SagittalSliceTag = mitkWidget2->GetSliceNavigationController()->AddObserver(mitk::SliceNavigationController::GeometrySliceEvent(NULL, 0), onSagittalSliceChangedCommand);

  itk::ReceptorMemberCommand<QmitkMIDASStdMultiWidget>::Pointer onCoronalSliceChangedCommand =
    itk::ReceptorMemberCommand<QmitkMIDASStdMultiWidget>::New();
  onCoronalSliceChangedCommand->SetCallbackFunction( this, &QmitkMIDASStdMultiWidget::OnCoronalSliceChanged );
  m_CoronalSliceTag = mitkWidget3->GetSliceNavigationController()->AddObserver(mitk::SliceNavigationController::GeometrySliceEvent(NULL, 0), onCoronalSliceChangedCommand);

}

QmitkMIDASStdMultiWidget::~QmitkMIDASStdMultiWidget()
{
  if (mitkWidget1 != NULL && m_AxialSliceTag != 0)
  {
    mitkWidget1->GetSliceNavigationController()->RemoveObserver(m_AxialSliceTag);
  }
  if (mitkWidget2 != NULL && m_SagittalSliceTag != 0)
  {
    mitkWidget2->GetSliceNavigationController()->RemoveObserver(m_SagittalSliceTag);
  }
  if (mitkWidget3 != NULL && m_CoronalSliceTag != 0)
  {
    mitkWidget3->GetSliceNavigationController()->RemoveObserver(m_CoronalSliceTag);
  }
}

void QmitkMIDASStdMultiWidget::OnNodesDropped(QmitkRenderWindow *window, std::vector<mitk::DataNode*> nodes)
{
  emit NodesDropped(this, window, nodes);
}

void QmitkMIDASStdMultiWidget::OnAxialSliceChanged(const itk::EventObject & geometrySliceEvent)
{
  this->OnPositionChanged();
}

void QmitkMIDASStdMultiWidget::OnSagittalSliceChanged(const itk::EventObject & geometrySliceEvent)
{
  this->OnPositionChanged();
}

void QmitkMIDASStdMultiWidget::OnCoronalSliceChanged(const itk::EventObject & geometrySliceEvent)
{
  this->OnPositionChanged();
}

void QmitkMIDASStdMultiWidget::OnPositionChanged()
{
  const mitk::Geometry3D *geometry = mitkWidget1->GetSliceNavigationController()->GetInputWorldGeometry();
  if (geometry != NULL)
  {
    mitk::Point3D voxelPoint;
    mitk::Point3D millimetrePoint = this->GetCrossPosition();

    geometry->WorldToIndex(millimetrePoint, voxelPoint);

    emit PositionChanged(voxelPoint, millimetrePoint);
  }
}

void QmitkMIDASStdMultiWidget::SetBackgroundColor(QColor colour)
{
  m_BackgroundColor = colour;
  m_GradientBackground1->SetGradientColors(colour.redF(),colour.greenF(),colour.blueF(),colour.redF(),colour.greenF(),colour.blueF());
  m_GradientBackground1->Enable();
  m_GradientBackground2->SetGradientColors(colour.redF(),colour.greenF(),colour.blueF(),colour.redF(),colour.greenF(),colour.blueF());
  m_GradientBackground2->Enable();
  m_GradientBackground3->SetGradientColors(colour.redF(),colour.greenF(),colour.blueF(),colour.redF(),colour.greenF(),colour.blueF());
  m_GradientBackground3->Enable();
  m_GradientBackground4->SetGradientColors(colour.redF(),colour.greenF(),colour.blueF(),colour.redF(),colour.greenF(),colour.blueF());
  m_GradientBackground4->Enable();
}

QColor QmitkMIDASStdMultiWidget::GetBackgroundColor() const
{
  return m_BackgroundColor;
}

void QmitkMIDASStdMultiWidget::SetSelected(bool b)
{
  m_IsSelected = b;
  if (b)
  {
    this->EnableColoredRectangles();
  }
  else
  {
    this->DisableColoredRectangles();
  }
}

bool QmitkMIDASStdMultiWidget::IsSelected() const
{
  return m_IsSelected;
}

void QmitkMIDASStdMultiWidget::SetSelectedWindow(vtkRenderWindow* window)
{
  // When we "Select", the selection is at the level of the QmitkMIDASStdMultiWidget
  // so the whole of this widget is selected. However, we may have clicked in
  // a specific view, so it still helps to highlight the most recently clicked on view.
  // Also, if you are displaying orthoview then you actually have 4 windows present,
  // then highlighting them all starts to look a bit confusing, so we just highlight the
  // most recently focussed window, (eg. axial, sagittal, coronal or 3D).

  if (this->ContainsVtkRenderWindow(window))
  {
    m_IsSelected = true;

    if (window == this->GetRenderWindow1()->GetVtkRenderWindow())
    {
      m_RectangleRendering1->Enable(1.0, 0.0, 0.0);
      m_RectangleRendering2->Disable();
      m_RectangleRendering3->Disable();
      m_RectangleRendering4->Disable();
    }
    else if (window == this->GetRenderWindow2()->GetVtkRenderWindow())
    {
      m_RectangleRendering1->Disable();
      m_RectangleRendering2->Enable(0.0, 1.0, 0.0);
      m_RectangleRendering3->Disable();
      m_RectangleRendering4->Disable();
    }
    else if (window == this->GetRenderWindow3()->GetVtkRenderWindow())
    {
      m_RectangleRendering1->Disable();
      m_RectangleRendering2->Disable();
      m_RectangleRendering3->Enable(0.0, 0.0, 1.0);
      m_RectangleRendering4->Disable();
    }
    else if (window == this->GetRenderWindow4()->GetVtkRenderWindow())
    {
      m_RectangleRendering1->Disable();
      m_RectangleRendering2->Disable();
      m_RectangleRendering3->Disable();
      m_RectangleRendering4->Enable(1.0, 1.0, 0.0);
    }
  }
  else
  {
    this->SetSelected(false);
  }
}

std::vector<QmitkRenderWindow*> QmitkMIDASStdMultiWidget::GetSelectedWindows() const
{

  std::vector<QmitkRenderWindow*> result;

  if (m_RectangleRendering1->IsEnabled())
  {
    result.push_back(this->GetRenderWindow1());
  }
  if (m_RectangleRendering2->IsEnabled())
  {
    result.push_back(this->GetRenderWindow2());
  }
  if (m_RectangleRendering3->IsEnabled())
  {
    result.push_back(this->GetRenderWindow3());
  }
  if (m_RectangleRendering4->IsEnabled())
  {
    result.push_back(this->GetRenderWindow4());
  }
  return result;
}

void QmitkMIDASStdMultiWidget::RequestUpdate()
{
  // The point of all this is to minimise the number of Updates.
  // So, ONLY call RequestUpdate on the specific window that is shown.

  if (this->isVisible() && this->IsEnabled())
  {
    switch(this->m_View)
    {
    case MIDAS_VIEW_AXIAL:
      m_RenderingManager->RequestUpdate(mitkWidget1->GetRenderWindow());
      break;
    case MIDAS_VIEW_SAGITTAL:
      m_RenderingManager->RequestUpdate(mitkWidget2->GetRenderWindow());
      break;
    case MIDAS_VIEW_CORONAL:
      m_RenderingManager->RequestUpdate(mitkWidget3->GetRenderWindow());
      break;
    case MIDAS_VIEW_ORTHO:
      m_RenderingManager->RequestUpdate(mitkWidget1->GetRenderWindow());
      m_RenderingManager->RequestUpdate(mitkWidget2->GetRenderWindow());
      m_RenderingManager->RequestUpdate(mitkWidget3->GetRenderWindow());
      m_RenderingManager->RequestUpdate(mitkWidget4->GetRenderWindow());
      break;
    case MIDAS_VIEW_3D:
      m_RenderingManager->RequestUpdate(mitkWidget4->GetRenderWindow());
      break;
    default:
      // die, this should never happen
      assert(m_View >= 0 && m_View <= 4);
      break;
    }
  }
}

void QmitkMIDASStdMultiWidget::SetEnabled(bool b)
{
  // See also constructor for things that are ALWAYS on/off.
  if (b && !m_IsEnabled)
  {
    this->AddPlanesToDataStorage();
  }
  else if (!b && m_IsEnabled)
  {
    this->RemovePlanesFromDataStorage();
  }
  m_IsEnabled = b;
}

bool QmitkMIDASStdMultiWidget::IsEnabled() const
{
  return m_IsEnabled;
}

void QmitkMIDASStdMultiWidget::SetDisplay2DCursorsLocally(bool visible)
{
  // Here, "locally" means, for this widget, so we are setting Renderer Specific properties.
  m_Display2DCursorsLocally = visible;
  this->SetVisibility(mitkWidget1, m_PlaneNode1, visible);
  this->SetVisibility(mitkWidget1, m_PlaneNode2, visible);
  this->SetVisibility(mitkWidget1, m_PlaneNode3, visible);
  this->SetVisibility(mitkWidget2, m_PlaneNode1, visible);
  this->SetVisibility(mitkWidget2, m_PlaneNode2, visible);
  this->SetVisibility(mitkWidget2, m_PlaneNode3, visible);
  this->SetVisibility(mitkWidget3, m_PlaneNode1, visible);
  this->SetVisibility(mitkWidget3, m_PlaneNode2, visible);
  this->SetVisibility(mitkWidget3, m_PlaneNode3, visible);
}

bool QmitkMIDASStdMultiWidget::GetDisplay2DCursorsLocally() const
{
  return m_Display2DCursorsLocally;
}

void QmitkMIDASStdMultiWidget::SetDisplay2DCursorsGlobally(bool visible)
{
  // Here, "globally" means the plane nodes created within this widget will be available in ALL other render windows.
  m_Display2DCursorsGlobally = visible;
  m_PlaneNode1->SetVisibility(visible);
  m_PlaneNode1->Modified();
  m_PlaneNode2->SetVisibility(visible);
  m_PlaneNode2->Modified();
  m_PlaneNode3->SetVisibility(visible);
  m_PlaneNode3->Modified();
}

bool QmitkMIDASStdMultiWidget::GetDisplay2DCursorsGlobally() const
{
  return m_Display2DCursorsGlobally;
}

void QmitkMIDASStdMultiWidget::SetDisplay3DViewInOrthoView(bool visible)
{
  m_Display3DViewInOrthoView = visible;
  this->Update3DWindowVisibility();
}

bool QmitkMIDASStdMultiWidget::GetDisplay3DViewInOrthoView() const
{
  return m_Display3DViewInOrthoView;
}

void QmitkMIDASStdMultiWidget::Update3DWindowVisibility()
{
  assert(m_DataStorage);

  vtkRenderWindow *axialVtkRenderWindow = this->mitkWidget1->GetVtkRenderWindow();
  mitk::BaseRenderer* axialRenderer = mitk::BaseRenderer::GetInstance(axialVtkRenderWindow);

  bool visibleIn3DWindow = false;
  if ((this->m_View == MIDAS_VIEW_ORTHO && this->m_Display3DViewInOrthoView)
     || this->m_View == MIDAS_VIEW_3D)
  {
    visibleIn3DWindow = true;
  }

  bool show3DPlanes = false;
  mitk::DataStorage::SetOfObjects::ConstPointer all = m_DataStorage->GetAll();
  for (mitk::DataStorage::SetOfObjects::ConstIterator it = all->Begin(); it != all->End(); ++it)
  {
    if (it->Value().IsNull())
    {
      continue;
    }

    bool isHelperNode = false;
    it->Value()->GetBoolProperty("helper object", isHelperNode);
    if (isHelperNode)
    {
      continue;
    }

    bool visibleInAxialView = false;
    if (it->Value()->GetBoolProperty("visible", visibleInAxialView, axialRenderer))
    {
      if (!visibleInAxialView)
      {
        visibleIn3DWindow = false;
      }
    }
    this->SetVisibility(this->mitkWidget4, it->Value(), visibleIn3DWindow);
    if (visibleIn3DWindow)
    {
      show3DPlanes = true;
    }
  }
  this->SetVisibility(this->mitkWidget4, m_PlaneNode1, show3DPlanes);
  this->SetVisibility(this->mitkWidget4, m_PlaneNode2, show3DPlanes);
  this->SetVisibility(this->mitkWidget4, m_PlaneNode3, show3DPlanes);
}

void QmitkMIDASStdMultiWidget::SetVisibility(QmitkRenderWindow *window, mitk::DataNode *node, bool visible)
{
  if (window != NULL && node != NULL)
  {
    vtkRenderWindow *vtkRenderWindow = window->GetVtkRenderWindow();
    if (vtkRenderWindow != NULL)
    {
      mitk::BaseRenderer* renderer = mitk::BaseRenderer::GetInstance(vtkRenderWindow);
      if (renderer != NULL)
      {
        bool currentVisibility = false;
        node->GetVisibility(currentVisibility, renderer);

        if (visible != currentVisibility)
        {
          node->SetVisibility(visible, renderer);
        }
      }
    }
  }
}

void QmitkMIDASStdMultiWidget::SetRendererSpecificVisibility(std::vector<mitk::DataNode*> nodes, bool visible)
{
  for (unsigned int i = 0; i < nodes.size(); i++)
  {
    this->SetVisibility(mitkWidget1, nodes[i], visible);
    this->SetVisibility(mitkWidget2, nodes[i], visible);
    this->SetVisibility(mitkWidget3, nodes[i], visible);
  }
  this->Update3DWindowVisibility();
}


bool QmitkMIDASStdMultiWidget::ContainsWindow(QmitkRenderWindow *window) const
{
  bool result = false;
  if (   mitkWidget1 == window
      || mitkWidget2 == window
      || mitkWidget3 == window
      || mitkWidget4 == window
      )
  {
    result = true;
  }
  return result;
}

bool QmitkMIDASStdMultiWidget::ContainsVtkRenderWindow(vtkRenderWindow *window) const
{
  bool result = false;
  if (   mitkWidget1->GetVtkRenderWindow() == window
      || mitkWidget2->GetVtkRenderWindow() == window
      || mitkWidget3->GetVtkRenderWindow() == window
      || mitkWidget4->GetVtkRenderWindow() == window
      )
  {
    result = true;
  }
  return result;
}

std::vector<QmitkRenderWindow*> QmitkMIDASStdMultiWidget::GetAllWindows() const
{
  std::vector<QmitkRenderWindow*> result;
  result.push_back(this->GetRenderWindow1());
  result.push_back(this->GetRenderWindow2());
  result.push_back(this->GetRenderWindow3());
  result.push_back(this->GetRenderWindow4());
  return result;
}

std::vector<vtkRenderWindow*> QmitkMIDASStdMultiWidget::GetAllVtkWindows() const
{
  std::vector<vtkRenderWindow*> result;
  result.push_back(this->GetRenderWindow1()->GetVtkRenderWindow());
  result.push_back(this->GetRenderWindow2()->GetVtkRenderWindow());
  result.push_back(this->GetRenderWindow3()->GetVtkRenderWindow());
  result.push_back(this->GetRenderWindow4()->GetVtkRenderWindow());
  return result;
}

MIDASOrientation QmitkMIDASStdMultiWidget::GetOrientation()
{
  MIDASOrientation result = MIDAS_ORIENTATION_UNKNOWN;
  if (m_View == MIDAS_VIEW_AXIAL)
  {
    result = MIDAS_ORIENTATION_AXIAL;
  }
  else if (m_View == MIDAS_VIEW_SAGITTAL)
  {
    result = MIDAS_ORIENTATION_SAGITTAL;
  }
  else if (m_View == MIDAS_VIEW_CORONAL)
  {
    result = MIDAS_ORIENTATION_CORONAL;
  }
  else if (m_View == MIDAS_VIEW_ORTHO)
  {
    if (m_RectangleRendering1->IsEnabled())
    {
      result = MIDAS_ORIENTATION_AXIAL;
    }
    else if (m_RectangleRendering2->IsEnabled())
    {
      result = MIDAS_ORIENTATION_SAGITTAL;
    }
    else if (m_RectangleRendering3->IsEnabled())
    {
      result = MIDAS_ORIENTATION_CORONAL;
    }
  }
  return result;
}

void QmitkMIDASStdMultiWidget::SetMIDASView(MIDASView view, mitk::Geometry3D *geometry)
{
  this->SetGeometry(geometry);
  this->SetMIDASView(view, false);
}

void QmitkMIDASStdMultiWidget::SetGeometry(mitk::Geometry3D *geometry)
{
  // Add these the first time we have a real geometry.
  if (geometry != NULL)
  {
    this->m_CornerAnnotaions[0].cornerText->SetText(0, "Axial");
    this->m_CornerAnnotaions[1].cornerText->SetText(0, "Sagittal");
    this->m_CornerAnnotaions[2].cornerText->SetText(0, "Coronal");

    std::vector< vtkRenderWindow* > registeredWindows = m_RenderingManager->GetAllRegisteredRenderWindows();
    if (registeredWindows.size() == 0)
    {
      m_RenderingManager->AddRenderWindow(this->GetRenderWindow1()->GetVtkRenderWindow());
      m_RenderingManager->AddRenderWindow(this->GetRenderWindow2()->GetVtkRenderWindow());
      m_RenderingManager->AddRenderWindow(this->GetRenderWindow3()->GetVtkRenderWindow());
      m_RenderingManager->AddRenderWindow(this->GetRenderWindow4()->GetVtkRenderWindow());
    }

    m_RenderingManager->InitializeViews(geometry);
  }
}

void QmitkMIDASStdMultiWidget::SetMIDASView(MIDASView view, bool rebuildLayout)
{
  if (rebuildLayout)
  {
    if (m_GridLayout != NULL)
    {
      delete m_GridLayout;
    }
    if (QmitkStdMultiWidgetLayout != NULL)
    {
      delete QmitkStdMultiWidgetLayout;
    }

    m_GridLayout = new QGridLayout();
    m_GridLayout->setContentsMargins(0, 0, 0, 0);
    m_GridLayout->setSpacing(0);

    QmitkStdMultiWidgetLayout = new QHBoxLayout( this );
    QmitkStdMultiWidgetLayout->setContentsMargins(0, 0, 0, 0);
    QmitkStdMultiWidgetLayout->setSpacing(0);

    m_GridLayout->addWidget(this->mitkWidget1Container, 0, 0);
    m_GridLayout->addWidget(this->mitkWidget2Container, 0, 1);
    m_GridLayout->addWidget(this->mitkWidget3Container, 1, 0);
    m_GridLayout->addWidget(this->mitkWidget4Container, 1, 1);

    QmitkStdMultiWidgetLayout->addLayout(m_GridLayout);
  }

  switch(view)
  {
  case MIDAS_VIEW_AXIAL:
    if (!this->mitkWidget1Container->isVisible())
    {
      this->mitkWidget1Container->show();
    }
    if (this->mitkWidget2Container->isVisible())
    {
      this->mitkWidget2Container->hide();
    }
    if (this->mitkWidget3Container->isVisible())
    {
      this->mitkWidget3Container->hide();
    }
    if (this->mitkWidget4Container->isVisible())
    {
      this->mitkWidget4Container->hide();
    }
    break;
  case MIDAS_VIEW_SAGITTAL:
    if (this->mitkWidget1Container->isVisible())
    {
      this->mitkWidget1Container->hide();
    }
    if (!this->mitkWidget2Container->isVisible())
    {
      this->mitkWidget2Container->show();
    }
    if (this->mitkWidget3Container->isVisible())
    {
      this->mitkWidget3Container->hide();
    }
    if (this->mitkWidget4Container->isVisible())
    {
      this->mitkWidget4Container->hide();
    }
    break;
  case MIDAS_VIEW_CORONAL:
    if (this->mitkWidget1Container->isVisible())
    {
      this->mitkWidget1Container->hide();
    }
    if (this->mitkWidget2Container->isVisible())
    {
      this->mitkWidget2Container->hide();
    }
    if (!this->mitkWidget3Container->isVisible())
    {
      this->mitkWidget3Container->show();
    }
    if (this->mitkWidget4Container->isVisible())
    {
      this->mitkWidget4Container->hide();
    }
    break;
  case MIDAS_VIEW_ORTHO:
    if (!this->mitkWidget1Container->isVisible())
    {
      this->mitkWidget1Container->show();
    }
    if (!this->mitkWidget2Container->isVisible())
    {
      this->mitkWidget2Container->show();
    }
    if (!this->mitkWidget3Container->isVisible())
    {
      this->mitkWidget3Container->show();
    }
    if (!this->mitkWidget4Container->isVisible())
    {
      this->mitkWidget4Container->show();
    }
    break;
  case MIDAS_VIEW_3D:
    if (this->mitkWidget1Container->isVisible())
    {
      this->mitkWidget1Container->hide();
    }
    if (this->mitkWidget2Container->isVisible())
    {
      this->mitkWidget2Container->hide();
    }
    if (this->mitkWidget3Container->isVisible())
    {
      this->mitkWidget3Container->hide();
    }
    if (!this->mitkWidget4Container->isVisible())
    {
      this->mitkWidget4Container->show();
    }
    break;
  default:
    // die, this should never happen
    assert(m_View >= 0 && m_View <= 4);
    break;
  }
  m_View = view;
  this->Update3DWindowVisibility();
  this->m_GridLayout->activate();
}

MIDASView QmitkMIDASStdMultiWidget::GetMIDASView() const
{
  return m_View;
}


bool QmitkMIDASStdMultiWidget::IsSingle2DView() const
{
  bool result = false;
  if (m_View == MIDAS_VIEW_AXIAL || m_View == MIDAS_VIEW_SAGITTAL || m_View == MIDAS_VIEW_CORONAL)
  {
    result = true;
  }
  return result;
}

mitk::SliceNavigationController::Pointer QmitkMIDASStdMultiWidget::GetSliceNavigationController(MIDASOrientation orientation) const
{
  mitk::SliceNavigationController::Pointer result = NULL;
  if (orientation == MIDAS_ORIENTATION_AXIAL)
  {
    result = mitkWidget1->GetSliceNavigationController();
  }
  else if (orientation == MIDAS_ORIENTATION_SAGITTAL)
  {
    result = mitkWidget2->GetSliceNavigationController();
  }
  else if (orientation == MIDAS_ORIENTATION_CORONAL)
  {
    result = mitkWidget3->GetSliceNavigationController();
  }
  return result;
}

unsigned int QmitkMIDASStdMultiWidget::GetMinSlice(MIDASOrientation orientation) const
{
  return 0;
}

unsigned int QmitkMIDASStdMultiWidget::GetMaxSlice(MIDASOrientation orientation) const
{
  unsigned int result = 0;

  mitk::SliceNavigationController::Pointer snc = this->GetSliceNavigationController(orientation);
  assert(snc);

  if (snc->GetSlice() != NULL)
  {
    if (snc->GetSlice()->GetSteps() >= 1)
    {
      result = snc->GetSlice()->GetSteps() -1;
    }
  }
  return result;
}

unsigned int QmitkMIDASStdMultiWidget::GetMinTime() const
{
  return 0;
}

unsigned int QmitkMIDASStdMultiWidget::GetMaxTime() const
{
  unsigned int result = 0;

  mitk::SliceNavigationController::Pointer snc = this->GetSliceNavigationController(MIDAS_ORIENTATION_AXIAL);
  assert(snc);

  if (snc->GetTime() != NULL)
  {
    if (snc->GetTime()->GetSteps() >= 1)
    {
      result = snc->GetTime()->GetSteps() -1;
    }
  }
  return result;
}

void QmitkMIDASStdMultiWidget::SetSliceNumber(MIDASOrientation orientation, unsigned int sliceNumber)
{
  mitk::SliceNavigationController::Pointer snc = this->GetSliceNavigationController(orientation);
  assert(snc);

  snc->GetSlice()->SetPos(sliceNumber);
}

unsigned int QmitkMIDASStdMultiWidget::GetSliceNumber(MIDASOrientation orientation) const
{
  mitk::SliceNavigationController::Pointer snc = this->GetSliceNavigationController(orientation);
  assert(snc);

  return snc->GetSlice()->GetPos();
}

void QmitkMIDASStdMultiWidget::SetTime(unsigned int timeSlice)
{
  mitk::SliceNavigationController::Pointer snc = this->GetSliceNavigationController(MIDAS_ORIENTATION_AXIAL);
  assert(snc);

  snc->GetTime()->SetPos(timeSlice);
}

unsigned int QmitkMIDASStdMultiWidget::GetTime() const
{
  mitk::SliceNavigationController::Pointer snc = this->GetSliceNavigationController(MIDAS_ORIENTATION_AXIAL);
  assert(snc);

  return snc->GetTime()->GetPos();
}

int QmitkMIDASStdMultiWidget::GetMagnificationFactor() const
{
  return m_MagnificationFactor;
}

void QmitkMIDASStdMultiWidget::SetMagnificationFactor(int magnificationFactor)
{
  // The aim of this method, is that when a magnificationFactor is passed in,
  // all 2D views update to an equivalent zoom, even if they were different beforehand.
  // The magnification factor is as it would be displayed in MIDAS, i.e. an integer
  // that corresponds to the rules given at the top of the header file.

  // Loop over axial, coronal, sagittal windows, the first 3 of 4 QmitkRenderWindow.
  std::vector<QmitkRenderWindow*> windows = this->GetAllWindows();
  for (unsigned int i = 0; i < 3; i++)
  {
    QmitkRenderWindow *window = windows[i];

    mitk::Point2D scaleFactorPixPerVoxel;
    mitk::Point2D scaleFactorPixPerMillimetres;
    this->GetScaleFactors(window, scaleFactorPixPerVoxel, scaleFactorPixPerMillimetres);

    double effectiveMagnificationFactor = 0;
    if (magnificationFactor >= 0)
    {
      effectiveMagnificationFactor = magnificationFactor + 1;
    }
    else
    {
      effectiveMagnificationFactor = magnificationFactor - 1;
      effectiveMagnificationFactor = fabs(1.0/effectiveMagnificationFactor);
    }

    mitk::Point2D targetScaleFactorPixPerMillimetres;

    // Need to scale both of the current scaleFactorPixPerVoxel[i]
    for (int i = 0; i < 2; i++)
    {
      targetScaleFactorPixPerMillimetres[i] = (effectiveMagnificationFactor / scaleFactorPixPerVoxel[i]) * scaleFactorPixPerMillimetres[i];
    }

    // Pick the one that has changed the least
    int axisWithLeastDifference = -1;
    double leastDifference = std::numeric_limits<double>::max();
    for(int i = 0; i < 2; i++)
    {
      if (fabs(targetScaleFactorPixPerMillimetres[i] - scaleFactorPixPerMillimetres[i]) < leastDifference)
      {
        leastDifference = fabs(targetScaleFactorPixPerMillimetres[i] - scaleFactorPixPerMillimetres[i]);
        axisWithLeastDifference = i;
      }
    }

    double zoomScaleFactor = targetScaleFactorPixPerMillimetres[axisWithLeastDifference]/scaleFactorPixPerMillimetres[axisWithLeastDifference];
    this->ZoomDisplayAboutCentre(window, zoomScaleFactor);

  } // end for

  m_MagnificationFactor = magnificationFactor;
  this->RequestUpdate();
}

int QmitkMIDASStdMultiWidget::FitMagnificationFactor()
{
  int magnificationFactor = 0;

  MIDASOrientation orientation = this->GetOrientation();
  if (orientation != MIDAS_ORIENTATION_UNKNOWN)
  {
    // Note, that this method should only be called for MIDAS purposes, when the view is a 2D
    // view, so it will either be Axial, Coronal, Sagittal, and not 3D or OthoView.

    QmitkRenderWindow *window = NULL;
    if (orientation == MIDAS_ORIENTATION_AXIAL)
    {
      window = this->GetRenderWindow1();
    }
    else if (orientation == MIDAS_ORIENTATION_SAGITTAL)
    {
      window = this->GetRenderWindow2();
    } else if (orientation == MIDAS_ORIENTATION_CORONAL)
    {
      window = this->GetRenderWindow3();
    }

    // Given the above comment, this means, we MUST have a window from this choice of 3.
    assert(window);

    //////////////////////////////////////////////////////////////////////////
    // Use the current window to work out a reasonable magnification factor.
    // This has to be done for each window, as they may be out of sync
    // due to the user manually (right click + mouse move) zooming the window.
    //////////////////////////////////////////////////////////////////////////

    // We do this with mitk::Point2D, so we have different values in X and Y, as images can be anisotropic.
    mitk::Point2D scaleFactorPixPerVoxel;
    mitk::Point2D scaleFactorPixPerMillimetres;
    this->GetScaleFactors(window, scaleFactorPixPerVoxel, scaleFactorPixPerMillimetres);

    // Now we scale these values so we get an integer number of pixels per voxel.
    mitk::Point2D targetScaleFactorPixPerVoxel;
    mitk::Point2D targetScaleFactorPixPerMillimetres;

    // Need to round the scale factors.
    for (int i = 0; i < 2; i++)
    {
      // If they are >= than 1, we round down towards 1
      // so you have less pixels per voxel, so image will appear smaller.
      if (scaleFactorPixPerVoxel[i] >= 1)
      {
        targetScaleFactorPixPerVoxel[i] = (int)(scaleFactorPixPerVoxel[i]);
      }
      else
      {
        // Otherwise, we still have to make image smaller to fit it on screen.
        //
        // For example, if the scale factor is 0.5, we invert it to get 2, which is an integer, so OK.
        // If however the scale factor is 0.4, we invert it to get 2.5 voxels per pixel, so we have
        // to round it up to 3, which means the image gets smaller (3 voxels fit into 1 pixel), and then
        // invert it to get the scale factor again.
        double tmp = 1.0 / scaleFactorPixPerVoxel[i];
        int roundedTmp = (int)(tmp + 0.5);
        tmp = 1.0 / (double) roundedTmp;
        targetScaleFactorPixPerVoxel[i] = tmp;
      }
      targetScaleFactorPixPerMillimetres[i] = scaleFactorPixPerMillimetres[i] * (targetScaleFactorPixPerVoxel[i]/scaleFactorPixPerVoxel[i]);
    }

    // We may have anisotropic voxels, so find the axis that requires most scale factor change.
    int axisWithLargestDifference = 0;
    double largestDifference = -1.0 * (std::numeric_limits<double>::max());
    for(int i = 0; i < 2; i++)
    {
      if (fabs(targetScaleFactorPixPerVoxel[i] - scaleFactorPixPerVoxel[i]) > largestDifference)
      {
        largestDifference = fabs(targetScaleFactorPixPerVoxel[i] - scaleFactorPixPerVoxel[i]);
        axisWithLargestDifference = i;
      }
    }

    /*
    * Note: The requirements specification for MIDAS style zoom basically says.
    *
    * magnification   : actual pixels per voxel.
    * on MIDAS widget :
    * 2               : 3
    * 1               : 2
    * 0               : 1 (i.e. no magnification).
    * -1              : 0.5 (i.e. 1 pixel covers 2 voxels).
    * -2              : 0.33 (i.e. 1 pixel covers 3 voxels).
    */

    // See comments at top of header file
    if (targetScaleFactorPixPerVoxel[axisWithLargestDifference] > 0)
    {
      // So, if pixels per voxel = 2, midas magnification = 1.
      // So, if pixels per voxel = 1, midas magnification = 0. etc.
      magnificationFactor = targetScaleFactorPixPerVoxel[axisWithLargestDifference] - 1;
    }
    else
    {
      magnificationFactor = (int)(1.0 / targetScaleFactorPixPerVoxel[axisWithLargestDifference]) + 1;
    }
  }
  return magnificationFactor;
}

void QmitkMIDASStdMultiWidget::GetScaleFactors(
    QmitkRenderWindow *window,
    mitk::Point2D &scaleFactorPixPerVoxel,
    mitk::Point2D &scaleFactorPixPerMillimetres)
{
  // Basic initialization - default value is arbitrarily 1 in both cases.
  scaleFactorPixPerVoxel[0] = 1;
  scaleFactorPixPerVoxel[1] = 1;
  scaleFactorPixPerMillimetres[0] = 1;
  scaleFactorPixPerMillimetres[1] = 1;

  if (window != NULL)
  {
    const mitk::Geometry3D *geometry = window->GetSliceNavigationController()->GetInputWorldGeometry();

    mitk::SliceNavigationController* sliceNavigationController = window->GetSliceNavigationController();
    assert(sliceNavigationController);

    mitk::BaseRenderer::Pointer baseRenderer = sliceNavigationController->GetRenderer();
    assert(baseRenderer);

    mitk::DisplayGeometry::Pointer displayGeometry = baseRenderer->GetDisplayGeometry();
    assert(displayGeometry);

    mitk::Point3D cornerPointsInImage[8];
    cornerPointsInImage[0] = geometry->GetCornerPoint(true, true, true);
    cornerPointsInImage[1] = geometry->GetCornerPoint(true, true, false);
    cornerPointsInImage[2] = geometry->GetCornerPoint(true, false, true);
    cornerPointsInImage[3] = geometry->GetCornerPoint(true, false, false);
    cornerPointsInImage[4] = geometry->GetCornerPoint(false, true, true);
    cornerPointsInImage[5] = geometry->GetCornerPoint(false, true, false);
    cornerPointsInImage[6] = geometry->GetCornerPoint(false, false, true);
    cornerPointsInImage[7] = geometry->GetCornerPoint(false, false, false);

    scaleFactorPixPerVoxel[0] = std::numeric_limits<float>::max();
    scaleFactorPixPerVoxel[1] = std::numeric_limits<float>::max();

    // Take every combination of pairs of 3D corner points taken from the 8 corners of the geometry.
    for (unsigned int i = 0; i < 8; i++)
    {
      mitk::Point3D pointsInVoxels[2];

      for (unsigned int j = 1; j < 8; j++)
      {
        geometry->WorldToIndex(cornerPointsInImage[i], pointsInVoxels[0]);
        geometry->WorldToIndex(cornerPointsInImage[j], pointsInVoxels[1]);

        // We only want to pick pairs of points where the points are different
        // and also differ in 3D space along exactly one axis (i.e. no diagonals).
        unsigned int differentVoxelIndexesCounter=0;
        int differentVoxelAxis = -1;

        for (unsigned int k = 0; k < 3; k++)
        {
          if (fabs(pointsInVoxels[1][k] - pointsInVoxels[0][k]) > 0.1)
          {
            differentVoxelIndexesCounter++;
            differentVoxelAxis = k;
          }
        }
        if (differentVoxelIndexesCounter == 1)
        {
          // So, for this pair (i,j) of points, project to 2D
          mitk::Point2D displayPointInMillimetreCoordinates[2];
          mitk::Point2D displayPointInPixelCoordinates[2];

          displayGeometry->Map(cornerPointsInImage[i], displayPointInMillimetreCoordinates[0]);
          displayGeometry->WorldToDisplay(displayPointInMillimetreCoordinates[0], displayPointInPixelCoordinates[0]);

          displayGeometry->Map(cornerPointsInImage[j], displayPointInMillimetreCoordinates[1]);
          displayGeometry->WorldToDisplay(displayPointInMillimetreCoordinates[1], displayPointInPixelCoordinates[1]);

          // Similarly, we only want to pick pairs of points where the projected 2D points
          // differ in 2D display coordinates along exactly one axis.
          unsigned int differentDisplayIndexesCounter=0;
          int differentDisplayAxis = -1;

          for (unsigned int k = 0; k < 2; k++)
          {
            if (fabs(displayPointInPixelCoordinates[1][k] - displayPointInPixelCoordinates[0][k]) > 0.1)
            {
              differentDisplayIndexesCounter++;
              differentDisplayAxis = k;
            }
          }
          if (differentDisplayIndexesCounter == 1)
          {
            // We now have i,j corresponding to a pair of points that are different in
            // 1 axis in voxel space, and different in one axis in diplay space, we can
            // use them to calculate scale factors.

            double distanceInMillimetres = cornerPointsInImage[i].EuclideanDistanceTo(cornerPointsInImage[j]);
            double distanceInVoxels = pointsInVoxels[0].EuclideanDistanceTo(pointsInVoxels[1]);
            double distanceInPixels = displayPointInPixelCoordinates[0].EuclideanDistanceTo(displayPointInPixelCoordinates[1]);
            double scaleFactorInDisplayPixelsPerImageVoxel = distanceInPixels / distanceInVoxels;
            double scaleFactorInDisplayPixelsPerMillimetres = distanceInPixels / distanceInMillimetres;

            if (scaleFactorInDisplayPixelsPerImageVoxel < scaleFactorPixPerVoxel[differentDisplayAxis])
            {
              scaleFactorPixPerVoxel[differentDisplayAxis] = scaleFactorInDisplayPixelsPerImageVoxel;
              scaleFactorPixPerMillimetres[differentDisplayAxis] = scaleFactorInDisplayPixelsPerMillimetres;
            }
          }
        }
      }
    }
  } // end window != NULL
}

void QmitkMIDASStdMultiWidget::ZoomDisplayAboutCentre(QmitkRenderWindow *window, double scaleFactor)
{
  if (window != NULL)
  {
    // I'm using assert statements, because fundamentally, if the window exists, so should all the other objects.
    mitk::SliceNavigationController* sliceNavigationController = window->GetSliceNavigationController();
    assert(sliceNavigationController);

    mitk::BaseRenderer* baseRenderer = sliceNavigationController->GetRenderer();
    assert(baseRenderer);

    mitk::DisplayGeometry* displayGeometry = baseRenderer->GetDisplayGeometry();
    assert(displayGeometry);

    mitk::Vector2D sizeInDisplayUnits = displayGeometry->GetSizeInDisplayUnits();
    mitk::Point2D centreOfDisplayInDisplayUnits;

    centreOfDisplayInDisplayUnits[0] = (sizeInDisplayUnits[0]-1.0)/2.0;
    centreOfDisplayInDisplayUnits[1] = (sizeInDisplayUnits[1]-1.0)/2.0;

    // Note that the scaleFactor is cumulative or multiplicative rather than absolute.
    displayGeometry->Zoom(scaleFactor, centreOfDisplayInDisplayUnits);
  }
}
