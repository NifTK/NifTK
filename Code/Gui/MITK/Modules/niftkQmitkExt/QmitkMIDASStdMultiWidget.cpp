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
#include "vtkSmartPointer.h"
#include "vtkMatrix4x4.h"
#include <QStackedLayout>
#include <QGridLayout>
#include <QFrame>
#include "itkConversionUtils.h"
#include "itkMatrix.h"
#include "itkSpatialOrientationAdapter.h"

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

  this->SetDataStorage(dataStorage);
  
  // We don't need these 4 lines if we pass in a widget specific RenderingManager.
  // If we are using a global one then we should use them to try and avoid Invalid Drawable errors on Mac.
  if (m_RenderingManager == mitk::RenderingManager::GetInstance())
  {
    m_RenderingManager->RemoveRenderWindow(this->mitkWidget1->GetVtkRenderWindow());
    m_RenderingManager->RemoveRenderWindow(this->mitkWidget2->GetVtkRenderWindow());
    m_RenderingManager->RemoveRenderWindow(this->mitkWidget3->GetVtkRenderWindow());
    m_RenderingManager->RemoveRenderWindow(this->mitkWidget4->GetVtkRenderWindow());
  }

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

  // Create 4 cameras for temporary storage.
  for (unsigned int i = 0; i < 4; i++)
  {
    m_Cameras[i] = vtkCamera::New();
  }

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

  for (unsigned int i = 0; i < 4; i++)
  {
    if (m_Cameras[i] != NULL)
    {
      m_Cameras[i]->Delete();
    }
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

  if (this->isVisible())
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

  bool show3DPlanes = false;

  mitk::DataStorage::SetOfObjects::ConstPointer all = m_DataStorage->GetAll();
  for (mitk::DataStorage::SetOfObjects::ConstIterator it = all->Begin(); it != all->End(); ++it)
  {
    if (it->Value().IsNull())
    {
      continue;
    }

    bool visibleIn3DWindow = false;
    if ((this->m_View == MIDAS_VIEW_ORTHO && this->m_Display3DViewInOrthoView)
        || this->m_View == MIDAS_VIEW_3D)
    {
      visibleIn3DWindow = true;
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
  // Add these annotations the first time we have a real geometry.
  if (geometry != NULL)
  {
    this->m_CornerAnnotaions[0].cornerText->SetText(0, "Axial");
    this->m_CornerAnnotaions[1].cornerText->SetText(0, "Sagittal");
    this->m_CornerAnnotaions[2].cornerText->SetText(0, "Coronal");

    // If m_RenderingManager is a local rendering manager
    // not the global singleton instance, then we never have to worry about this.
    if (m_RenderingManager == mitk::RenderingManager::GetInstance())
    {
      m_RenderingManager->AddRenderWindow(this->GetRenderWindow1()->GetVtkRenderWindow());
      m_RenderingManager->AddRenderWindow(this->GetRenderWindow2()->GetVtkRenderWindow());
      m_RenderingManager->AddRenderWindow(this->GetRenderWindow3()->GetVtkRenderWindow());
      m_RenderingManager->AddRenderWindow(this->GetRenderWindow4()->GetVtkRenderWindow());
    }

    std::vector<vtkRenderWindow*> vtkWindows = this->GetAllVtkWindows();
    for (unsigned int window = 0; window < vtkWindows.size(); window++)
    {
      vtkRenderWindow* vtkWindow = vtkWindows[window];
      mitk::BaseRenderer *baseRenderer = mitk::BaseRenderer::GetInstance(vtkWindow);
      int id = baseRenderer->GetMapperID();

      // Get access to slice navigation controller, as this sorts out most of the process.
      mitk::SliceNavigationController *sliceNavigationController = baseRenderer->GetSliceNavigationController();
      sliceNavigationController->SetViewDirectionToDefault();

      if (window < 3)
      {
        // Get the view/orientation flags.
        mitk::SliceNavigationController::ViewDirection viewDirection = sliceNavigationController->GetViewDirection();

        // Inspired by:
        // http://www.na-mic.org/Wiki/index.php/Coordinate_System_Conversion_Between_ITK_and_Slicer3

        mitk::AffineTransform3D::Pointer affineTransform = geometry->GetIndexToWorldTransform();
        itk::Matrix<float, 3, 3> affineTransformMatrix = affineTransform->GetMatrix();
        mitk::AffineTransform3D::MatrixType::InternalMatrixType normalisedAffineTransformMatrix;
        for (unsigned int i=0; i < 3; i++)
        {
          for (unsigned int j = 0; j < 3; j++)
          {
            normalisedAffineTransformMatrix[i][j] = affineTransformMatrix[i][j];
          }
        }
        normalisedAffineTransformMatrix.normalize_columns();
        for (unsigned int i=0; i < 3; i++)
        {
          for (unsigned int j = 0; j < 3; j++)
          {
            affineTransformMatrix[i][j] = normalisedAffineTransformMatrix[i][j];
          }
        }

        MITK_DEBUG << "Matt, original matrix=" << std::endl;

        for (unsigned int i=0; i < 3; i++)
        {
          for (unsigned int j = 0; j < 3; j++)
          {
            MITK_DEBUG << affineTransformMatrix[i][j] << " ";
          }
          MITK_DEBUG << std::endl;
        }

        mitk::AffineTransform3D::MatrixType::InternalMatrixType inverseTransformMatrix = affineTransformMatrix.GetInverse();

        MITK_DEBUG << "Matt, inverseTransformMatrix matrix=" << std::endl;

        for (unsigned int i=0; i < 3; i++)
        {
          for (unsigned int j = 0; j < 3; j++)
          {
            MITK_DEBUG << inverseTransformMatrix[i][j] << " ";
          }
          MITK_DEBUG << std::endl;
        }

        int dominantAxisRL = itk::Function::Max3(inverseTransformMatrix[0][0],inverseTransformMatrix[1][0],inverseTransformMatrix[2][0]);
        int signRL = itk::Function::Sign(inverseTransformMatrix[dominantAxisRL][0]);
        int dominantAxisAP = itk::Function::Max3(inverseTransformMatrix[0][1],inverseTransformMatrix[1][1],inverseTransformMatrix[2][1]);
        int signAP = itk::Function::Sign(inverseTransformMatrix[dominantAxisAP][1]);
        int dominantAxisSI = itk::Function::Max3(inverseTransformMatrix[0][2],inverseTransformMatrix[1][2],inverseTransformMatrix[2][2]);
        int signSI = itk::Function::Sign(inverseTransformMatrix[dominantAxisSI][2]);

        int permutedBoundingBox[3];
        int permutedAxes[3];
        int flippedAxes[3];
        double permutedSpacing[3];

        permutedAxes[0] = dominantAxisRL;
        permutedAxes[1] = dominantAxisAP;
        permutedAxes[2] = dominantAxisSI;

        flippedAxes[0] = signRL;
        flippedAxes[1] = signAP;
        flippedAxes[2] = signSI;

        permutedBoundingBox[0] = geometry->GetExtent(dominantAxisRL);
        permutedBoundingBox[1] = geometry->GetExtent(dominantAxisAP);
        permutedBoundingBox[2] = geometry->GetExtent(dominantAxisSI);

        permutedSpacing[0] = geometry->GetSpacing()[permutedAxes[0]];
        permutedSpacing[1] = geometry->GetSpacing()[permutedAxes[1]];
        permutedSpacing[2] = geometry->GetSpacing()[permutedAxes[2]];

        MITK_DEBUG << "Matt, domRL=" << dominantAxisRL << ", signRL=" << signRL << ", domAP=" << dominantAxisAP << ", signAP=" << signAP << ", dominantAxisSI=" << dominantAxisSI << ", signSI=" << signSI << std::endl;
        MITK_DEBUG << "Matt, permutedBoundingBox=" << permutedBoundingBox[0] << ", " << permutedBoundingBox[1] << ", " << permutedBoundingBox[2] << std::endl;
        MITK_DEBUG << "Matt, permutedAxes=" << permutedAxes[0] << ", " << permutedAxes[1] << ", " << permutedAxes[2] << std::endl;
        MITK_DEBUG << "Matt, permutedSpacing=" << permutedSpacing[0] << ", " << permutedSpacing[1] << ", " << permutedSpacing[2] << std::endl;
        MITK_DEBUG << "Matt, flippedAxes=" << flippedAxes[0] << ", " << flippedAxes[1] << ", " << flippedAxes[2] << std::endl;

        mitk::AffineTransform3D::MatrixType::InternalMatrixType permutedMatrix;
        permutedMatrix.set_column(0, inverseTransformMatrix.get_row(permutedAxes[0]) * flippedAxes[0]);
        permutedMatrix.set_column(1, inverseTransformMatrix.get_row(permutedAxes[1]) * flippedAxes[1]);
        permutedMatrix.set_column(2, inverseTransformMatrix.get_row(permutedAxes[2]) * flippedAxes[2]);

        MITK_DEBUG << "Matt, permuted matrix=" << std::endl;

        for (unsigned int i=0; i < 3; i++)
        {
          for (unsigned int j = 0; j < 3; j++)
          {
            MITK_DEBUG << permutedMatrix[i][j] << " ";
          }
          MITK_DEBUG << std::endl;
        }

        // Work out transformed voxel origin.
        mitk::Point3D originVoxels;
        mitk::Point3D originMillimetres;
        mitk::Point3D originOfSlice;

        originVoxels[0] = 0;
        originVoxels[1] = 0;
        originVoxels[2] = 0;

        MITK_DEBUG << "Matt, geometry->GetExtent(n)=" << geometry->GetExtent(0) << ", " << geometry->GetExtent(1) << ", " << geometry->GetExtent(2) << std::endl;

        if (flippedAxes[0] < 0) originVoxels[permutedAxes[0]] = geometry->GetExtent(permutedAxes[0]) - 1;
        if (flippedAxes[1] < 0) originVoxels[permutedAxes[1]] = geometry->GetExtent(permutedAxes[1]) - 1;
        if (flippedAxes[2] < 0) originVoxels[permutedAxes[2]] = geometry->GetExtent(permutedAxes[2]) - 1;

        geometry->IndexToWorld(originVoxels, originMillimetres);
        MITK_DEBUG << "Matt, RAI originVoxels=" << originVoxels << ", originMillimetres=" << originMillimetres << std::endl;

        if(geometry->GetImageGeometry())
        {
          originMillimetres[0] -= 0.5;
          originMillimetres[1] -= 0.5;
          originMillimetres[2] -= 0.5;
        }
        MITK_DEBUG << "Matt, RAI originVoxels=" << originVoxels << ", originMillimetres=" << originMillimetres << std::endl;

        // Setting up the width, height, axis orientation.
        mitk::VnlVector  rightDV(3);
        mitk::VnlVector  bottomDV(3);
        mitk::VnlVector  normal(3);
        int              width = 1;
        int              height = 1;
        mitk::ScalarType viewSpacing = 1;
        unsigned int     slices = 1;
        bool             isFlipped;

        switch(viewDirection)
        {
        case mitk::SliceNavigationController::Sagittal:
          width  = permutedBoundingBox[1];
          height = permutedBoundingBox[2];
          originOfSlice[0] = originMillimetres[0];
          originOfSlice[1] = originMillimetres[1];
          originOfSlice[2] = originMillimetres[2];
          rightDV[0] = permutedMatrix[0][1];
          rightDV[1] = permutedMatrix[1][1];
          rightDV[2] = permutedMatrix[2][1];
          bottomDV[0] = permutedMatrix[0][2];
          bottomDV[1] = permutedMatrix[1][2];
          bottomDV[2] = permutedMatrix[2][2];
          normal[0] = permutedMatrix[0][0];
          normal[1] = permutedMatrix[1][0];
          normal[2] = permutedMatrix[2][0];
          viewSpacing = permutedSpacing[0];
          slices = permutedBoundingBox[0];
          isFlipped = false;
          break;
        case mitk::SliceNavigationController::Frontal:
          width  = permutedBoundingBox[0];
          height = permutedBoundingBox[2];
          originOfSlice[0] = originMillimetres[0];
          originOfSlice[1] = originMillimetres[1];
          originOfSlice[2] = originMillimetres[2];
          rightDV[0] = permutedMatrix[0][0];
          rightDV[1] = permutedMatrix[1][0];
          rightDV[2] = permutedMatrix[2][0];
          bottomDV[0] = permutedMatrix[0][2];
          bottomDV[1] = permutedMatrix[1][2];
          bottomDV[2] = permutedMatrix[2][2];
          normal[0] = permutedMatrix[0][1];
          normal[1] = permutedMatrix[1][1];
          normal[2] = permutedMatrix[2][1];
          viewSpacing = permutedSpacing[1];
          slices = permutedBoundingBox[1];
          isFlipped = true;
          break;
        default:
          // Default = Transversal.
          width  = permutedBoundingBox[0];
          height = permutedBoundingBox[1];
          originOfSlice[0] = originMillimetres[0] + permutedBoundingBox[0]*permutedSpacing[0]*permutedMatrix[0][1];
          originOfSlice[1] = originMillimetres[1] + permutedBoundingBox[1]*permutedSpacing[1]*permutedMatrix[1][1];
          originOfSlice[2] = originMillimetres[2] + permutedBoundingBox[2]*permutedSpacing[2]*permutedMatrix[2][1];
          rightDV[0] = permutedMatrix[0][0];
          rightDV[1] = permutedMatrix[1][0];
          rightDV[2] = permutedMatrix[2][0];
          bottomDV[0] = -1.0 * permutedMatrix[0][1];
          bottomDV[1] = -1.0 * permutedMatrix[1][1];
          bottomDV[2] = -1.0 * permutedMatrix[2][1];
          normal[0] = permutedMatrix[0][2];
          normal[1] = permutedMatrix[1][2];
          normal[2] = permutedMatrix[2][2];
          viewSpacing = permutedSpacing[2];
          slices = permutedBoundingBox[2];
          isFlipped = true;
          break;
        }

        MITK_DEBUG << "Matt, width=" << width << std::endl;
        MITK_DEBUG << "Matt, height=" << height << std::endl;
        MITK_DEBUG << "Matt, originOfSlice=" << originOfSlice << std::endl;
        MITK_DEBUG << "Matt, rightDV=" << rightDV << std::endl;
        MITK_DEBUG << "Matt, bottomDV=" << bottomDV << std::endl;
        MITK_DEBUG << "Matt, normal=" << normal << std::endl;
        MITK_DEBUG << "Matt, viewSpacing=" << viewSpacing << std::endl;
        MITK_DEBUG << "Matt, slices=" << slices << std::endl;
        MITK_DEBUG << "Matt, isFlipped=" << isFlipped << std::endl;

        mitk::ScalarType bounds[6]= { 0, width, 0, height, 0, 1 };

        // A SlicedGeometry3D is initialised from a 2D plane geometry, plus the number of slices.
        // So here is the plane.
        mitk::PlaneGeometry::Pointer planeGeometry = mitk::PlaneGeometry::New();
        planeGeometry->SetIdentity();
        planeGeometry->SetImageGeometry(geometry->GetImageGeometry());
        planeGeometry->SetBounds(bounds);
        planeGeometry->SetMatrixByVectors(rightDV, bottomDV, normal.two_norm());
        planeGeometry->SetOrigin(originOfSlice);

        // And here we create the slicedGeometry from an initial plane, and a given number of slices.
        mitk::SlicedGeometry3D::Pointer slicedGeometry = mitk::SlicedGeometry3D::New();
        slicedGeometry->SetReferenceGeometry(geometry);
        slicedGeometry->SetImageGeometry(geometry->GetImageGeometry());
        slicedGeometry->InitializeEvenlySpaced(planeGeometry, viewSpacing, slices, isFlipped );

        // If input geometry is time sliced and we have >1 timestep, we must also cope with that.
        mitk::TimeSlicedGeometry::Pointer timeSlicedGeometry = static_cast<mitk::TimeSlicedGeometry*>(geometry);
        if (timeSlicedGeometry.IsNotNull() && timeSlicedGeometry->GetTimeSteps() > 1)
        {
          mitk::TimeSlicedGeometry::Pointer createdTimeSlicedGeometry = mitk::TimeSlicedGeometry::New();
          createdTimeSlicedGeometry->InitializeEmpty(timeSlicedGeometry->GetTimeSteps());
          createdTimeSlicedGeometry->SetImageGeometry(timeSlicedGeometry->GetImageGeometry());
          createdTimeSlicedGeometry->SetTimeBounds(timeSlicedGeometry->GetTimeBounds());
          createdTimeSlicedGeometry->SetEvenlyTimed(timeSlicedGeometry->GetEvenlyTimed());
          createdTimeSlicedGeometry->SetIndexToWorldTransform(slicedGeometry->GetIndexToWorldTransform());
          createdTimeSlicedGeometry->SetBounds(timeSlicedGeometry->GetBounds());

          for (unsigned int i = 0; i < timeSlicedGeometry->GetTimeSteps(); i++)
          {
            mitk::SlicedGeometry3D::Pointer createdSlicedGeometryForEachTimeStep = mitk::SlicedGeometry3D::New();
            createdSlicedGeometryForEachTimeStep->SetIdentity();
            createdSlicedGeometryForEachTimeStep->SetReferenceGeometry(geometry);
            createdSlicedGeometryForEachTimeStep->SetImageGeometry(geometry->GetImageGeometry());
            createdSlicedGeometryForEachTimeStep->InitializeEvenlySpaced(planeGeometry, viewSpacing, slices, isFlipped );
            createdSlicedGeometryForEachTimeStep->SetTimeBounds(timeSlicedGeometry->GetGeometry3D(i)->GetTimeBounds());

            createdTimeSlicedGeometry->SetGeometry3D(createdSlicedGeometryForEachTimeStep, i);
          }
          createdTimeSlicedGeometry->UpdateInformation();
          sliceNavigationController->SetInputWorldGeometry(createdTimeSlicedGeometry);
        }
        else
        {
          sliceNavigationController->SetInputWorldGeometry(slicedGeometry);
        }
        sliceNavigationController->Update(mitk::SliceNavigationController::Original, true, true, false);
      }

      // For 2D mappers only, set to middle slice (the 3D mapper simply follows by event listening).
      if ( id == 1 )
      {
        // Now geometry is established, set to middle slice.
        sliceNavigationController->GetSlice()->SetPos( sliceNavigationController->GetSlice()->GetSteps() / 2 );
      }
      // Now geometry is established, get the display geometry to fit the picture to the window.
      baseRenderer->GetDisplayGeometry()->SetConstrainZoomingAndPanning(false);
      baseRenderer->GetDisplayGeometry()->Fit();
    }
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
  snc->GetTime()->SetPos(timeSlice);

  snc = this->GetSliceNavigationController(MIDAS_ORIENTATION_SAGITTAL);
  snc->GetTime()->SetPos(timeSlice);

  snc = this->GetSliceNavigationController(MIDAS_ORIENTATION_CORONAL);
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

        for (unsigned int k = 0; k < 3; k++)
        {
          if (fabs(pointsInVoxels[1][k] - pointsInVoxels[0][k]) > 0.1)
          {
            differentVoxelIndexesCounter++;
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

void QmitkMIDASStdMultiWidget::StoreCameras()
{
  std::vector<QmitkRenderWindow*> windows = this->GetAllWindows();
  for (unsigned int i = 0; i < windows.size(); i++)
  {
    vtkCamera* camera = windows[i]->GetRenderer()->GetVtkRenderer()->GetActiveCamera();
    this->m_Cameras[i]->SetPosition(camera->GetPosition());
    this->m_Cameras[i]->SetFocalPoint(camera->GetFocalPoint());
    this->m_Cameras[i]->SetViewUp(camera->GetViewUp());
    this->m_Cameras[i]->SetClippingRange(camera->GetClippingRange());
  }
}

void QmitkMIDASStdMultiWidget::RestoreCameras()
{
  std::vector<QmitkRenderWindow*> windows = this->GetAllWindows();
  for (unsigned int i = 0; i < windows.size(); i++)
  {
    vtkCamera* camera = windows[i]->GetRenderer()->GetVtkRenderer()->GetActiveCamera();
    camera->SetPosition(this->m_Cameras[i]->GetPosition());
    camera->SetFocalPoint(this->m_Cameras[i]->GetFocalPoint());
    camera->SetViewUp(this->m_Cameras[i]->GetViewUp());
    camera->SetClippingRange(this->m_Cameras[i]->GetClippingRange());
  }
}

