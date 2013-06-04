/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkMIDASSegmentationViewWidget.h"
#include "QmitkMIDASBaseSegmentationFunctionality.h"
#include <QmitkMIDASSingleViewWidgetListVisibilityManager.h>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QSpacerItem>
#include <mitkDataStorage.h>
#include <mitkTimeSlicedGeometry.h>
#include <mitkGlobalInteraction.h>
#include <mitkFocusManager.h>
#include <mitkGeometry3D.h>
#include <mitkTimeSlicedGeometry.h>
#include <mitkSliceNavigationController.h>
#include <mitkBaseRenderer.h>
#include <itkCommand.h>


//-----------------------------------------------------------------------------
QmitkMIDASSegmentationViewWidget::QmitkMIDASSegmentationViewWidget(QWidget *parent)
: m_ContainingFunctionality(NULL)
, m_FocusManagerObserverTag(0)
, m_View(MIDAS_VIEW_UNKNOWN)
, m_MainWindowView(MIDAS_VIEW_UNKNOWN)
, m_MainWindowAxial(NULL)
, m_MainWindowSagittal(NULL)
, m_MainWindowCoronal(NULL)
, m_MainWindow3d(NULL)
, m_CurrentRenderer(NULL)
, m_NodeAddedSetter(NULL)
, m_VisibilityTracker(NULL)
{
  this->setupUi(parent);
  m_ViewerWidget->SetRememberViewSettingsPerOrientation(false);
  m_ViewerWidget->SetSelected(false);
  m_TwoViewCheckBox->setChecked(false);
  m_VerticalCheckBox->setChecked(false);
  m_VerticalCheckBox->setEnabled(false);
  m_AxialRadioButton->setChecked(true);
  ChangeLayout(true);

  std::vector<mitk::BaseRenderer*> renderers;
  renderers.push_back(m_ViewerWidget->GetAxialWindow()->GetRenderer());
  renderers.push_back(m_ViewerWidget->GetSagittalWindow()->GetRenderer());
  renderers.push_back(m_ViewerWidget->GetCoronalWindow()->GetRenderer());

  m_NodeAddedSetter = mitk::MIDASNodeAddedVisibilitySetter::New();
  m_NodeAddedSetter->SetRenderers(renderers);
  m_NodeAddedSetter->SetVisibility(false);

  m_VisibilityTracker = mitk::DataStorageVisibilityTracker::New();
  m_VisibilityTracker->SetRenderersToUpdate(renderers);

  m_ViewerWidget->SetDisplayInteractionEnabled(true);

  connect(m_TwoViewCheckBox, SIGNAL(stateChanged(int)), this, SLOT(OnTwoViewStateChanged(int)));
  connect(m_VerticalCheckBox, SIGNAL(stateChanged(int)), this, SLOT(OnVerticalLayoutStateChanged(int)));
  connect(m_AxialRadioButton, SIGNAL(toggled(bool)), this, SLOT(OnAxialToggled(bool)));
  connect(m_SagittalRadioButton, SIGNAL(toggled(bool)), this, SLOT(OnSagittalToggled(bool)));
  connect(m_CoronalRadioButton, SIGNAL(toggled(bool)), this, SLOT(OnCoronalToggled(bool)));
  connect(m_OrthoRadioButton, SIGNAL(toggled(bool)), this, SLOT(OnOrthoToggled(bool)));
}


//-----------------------------------------------------------------------------
QmitkMIDASSegmentationViewWidget::~QmitkMIDASSegmentationViewWidget()
{
  // m_NodeAddedSetter deleted by smart pointer.
  // m_VisibilityTracker deleted by smart pointer.
}


//-----------------------------------------------------------------------------
void QmitkMIDASSegmentationViewWidget::SetDataStorage(mitk::DataStorage* storage)
{
  if (storage != NULL)
  {
    m_ViewerWidget->SetDataStorage(storage);

    m_NodeAddedSetter->SetDataStorage(storage);
    m_VisibilityTracker->SetDataStorage(storage);
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASSegmentationViewWidget::SetContainingFunctionality(QmitkMIDASBaseSegmentationFunctionality* functionality)
{
  m_ContainingFunctionality = functionality;
}


//-----------------------------------------------------------------------------
void QmitkMIDASSegmentationViewWidget::Activated()
{
  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  if (focusManager != NULL)
  {
    itk::SimpleMemberCommand<QmitkMIDASSegmentationViewWidget>::Pointer onFocusChangedCommand =
      itk::SimpleMemberCommand<QmitkMIDASSegmentationViewWidget>::New();
    onFocusChangedCommand->SetCallbackFunction( this, &QmitkMIDASSegmentationViewWidget::OnFocusChanged );

    m_FocusManagerObserverTag = focusManager->AddObserver(mitk::FocusEvent(), onFocusChangedCommand);

    // Force this because:
    // If user launches Drag and Drop Display, loads image, drops into window, and THEN launches this widget,
    // you don't get the first OnFocusChanged as the window has already been clicked on, and then the
    // geometry is initialised incorrectly.
    this->OnFocusChanged();
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASSegmentationViewWidget::Deactivated()
{
  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  if (focusManager != NULL)
  {
    focusManager->RemoveObserver(m_FocusManagerObserverTag);
  }
  m_ViewerWidget->SetEnabled(false);
}


//-----------------------------------------------------------------------------
void QmitkMIDASSegmentationViewWidget::SetBlockSignals(bool block)
{
  m_AxialRadioButton->blockSignals(block);
  m_SagittalRadioButton->blockSignals(block);
  m_CoronalRadioButton->blockSignals(block);
  m_OrthoRadioButton->blockSignals(block);
  m_TwoViewCheckBox->blockSignals(block);
  m_VerticalCheckBox->blockSignals(block);
}


//-----------------------------------------------------------------------------
void QmitkMIDASSegmentationViewWidget::SetEnabled(bool enabled)
{
  this->EnableOrientationWidgets(enabled);
  m_TwoViewCheckBox->setEnabled(enabled);
  m_VerticalCheckBox->setEnabled(enabled);
}


//-----------------------------------------------------------------------------
void QmitkMIDASSegmentationViewWidget::EnableOrientationWidgets(bool enabled)
{
  m_AxialRadioButton->setEnabled(enabled);
  m_SagittalRadioButton->setEnabled(enabled);
  m_CoronalRadioButton->setEnabled(enabled);
  m_OrthoRadioButton->setEnabled(enabled);
}


//-----------------------------------------------------------------------------
void QmitkMIDASSegmentationViewWidget::OnTwoViewStateChanged(int state)
{
  this->ChangeLayout();
}


//-----------------------------------------------------------------------------
void QmitkMIDASSegmentationViewWidget::OnVerticalLayoutStateChanged(int state)
{
  this->ChangeLayout();
}


//-----------------------------------------------------------------------------
void QmitkMIDASSegmentationViewWidget::OnAxialToggled(bool)
{
  if (m_AxialRadioButton->isChecked())
  {
    this->ChangeLayout();
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASSegmentationViewWidget::OnCoronalToggled(bool)
{
  if (m_CoronalRadioButton->isChecked())
  {
    this->ChangeLayout();
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASSegmentationViewWidget::OnSagittalToggled(bool)
{
  if (m_SagittalRadioButton->isChecked())
  {
    this->ChangeLayout();
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASSegmentationViewWidget::OnOrthoToggled(bool)
{
  if (m_OrthoRadioButton->isChecked())
  {
    this->ChangeLayout();
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASSegmentationViewWidget::ChangeLayout(bool isInitialising)
{
  MIDASView nextView = MIDAS_VIEW_UNKNOWN;

  this->SetBlockSignals(true);

  if (m_TwoViewCheckBox->isChecked())
  {    
    if (m_VerticalCheckBox->isChecked())
    {
      if (m_MainWindowView == MIDAS_VIEW_AXIAL)
      {
        nextView = MIDAS_VIEW_COR_SAG_V;
      }
      else if (m_MainWindowView == MIDAS_VIEW_SAGITTAL)
      {
        nextView = MIDAS_VIEW_COR_AX_V;
      }
      else if (m_MainWindowView == MIDAS_VIEW_CORONAL)
      {
        nextView = MIDAS_VIEW_SAG_AX_V;
      }
    }
    else
    {
      if (m_MainWindowView == MIDAS_VIEW_AXIAL)
      {
        nextView = MIDAS_VIEW_COR_SAG_H;
      }
      else if (m_MainWindowView == MIDAS_VIEW_SAGITTAL)
      {
        nextView = MIDAS_VIEW_COR_AX_H;
      }
      else if (m_MainWindowView == MIDAS_VIEW_CORONAL)
      {
        nextView = MIDAS_VIEW_SAG_AX_H;
      }
    }
  }
  else
  {
    if (m_MainWindowView == MIDAS_VIEW_AXIAL)
    {
      if (m_SagittalRadioButton->isChecked())
      {
        nextView = MIDAS_VIEW_SAGITTAL;
      }
      else if (m_OrthoRadioButton->isChecked())
      {
        nextView = MIDAS_VIEW_ORTHO;
      }
      else
      {
        nextView = MIDAS_VIEW_CORONAL;
        m_CoronalRadioButton->setChecked(true);
      }
    }
    else if (m_MainWindowView == MIDAS_VIEW_SAGITTAL)
    {
      if (m_AxialRadioButton->isChecked())
      {
        nextView = MIDAS_VIEW_AXIAL;
      }
      else if (m_OrthoRadioButton->isChecked())
      {
        nextView = MIDAS_VIEW_ORTHO;
      }
      else
      {
        nextView = MIDAS_VIEW_CORONAL;
        m_CoronalRadioButton->setChecked(true);
      }
    }
    else if (m_MainWindowView == MIDAS_VIEW_CORONAL)
    {
      if (m_SagittalRadioButton->isChecked())
      {
        nextView = MIDAS_VIEW_SAGITTAL;
      }
      else if (m_OrthoRadioButton->isChecked())
      {
        nextView = MIDAS_VIEW_ORTHO;
      }
      else
      {
        nextView = MIDAS_VIEW_AXIAL;
        m_AxialRadioButton->setChecked(true);
      }
    }
  }

  this->EnableWidgets();
  this->SetBlockSignals(false);

  if (nextView != MIDAS_VIEW_UNKNOWN && nextView != m_View)
  {
    m_View = nextView;
    if (isInitialising)
    {
      m_ViewerWidget->SetView(m_View, false);
    }
    else
    {
      m_ViewerWidget->SwitchView(m_View);
    }
    emit ViewChanged(m_View);
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASSegmentationViewWidget::EnableWidgets()
{
  if (m_TwoViewCheckBox->isChecked())
  {
    m_VerticalCheckBox->setEnabled(true);
    this->EnableOrientationWidgets(false);
  }
  else
  {
    m_VerticalCheckBox->setEnabled(false);
    this->EnableOrientationWidgets(true);

    if (m_MainWindowView == MIDAS_VIEW_AXIAL)
    {
      m_AxialRadioButton->setEnabled(false);
    }
    else if (m_MainWindowView == MIDAS_VIEW_SAGITTAL)
    {
      m_SagittalRadioButton->setEnabled(false);
    }
    else if (m_MainWindowView == MIDAS_VIEW_CORONAL)
    {
      m_CoronalRadioButton->setEnabled(false);
    }
    else if (m_MainWindowView == MIDAS_VIEW_ORTHO)
    {
      m_OrthoRadioButton->setEnabled(false);
    }
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASSegmentationViewWidget::OnFocusChanged()
{
  // If the newly focused window is this widget, nothing to update. Stop early.
  if (this->IsCurrentlyFocusedWindowInThisWidget())
  {
    return;
  }

  // Get hold of main windows, using QmitkAbstractView lookup mitkIRenderWindowPart.
  QmitkRenderWindow *mainWindowAxial = m_ContainingFunctionality->GetRenderWindow("axial");
  QmitkRenderWindow *mainWindowSagittal = m_ContainingFunctionality->GetRenderWindow("sagittal");
  QmitkRenderWindow *mainWindowCoronal = m_ContainingFunctionality->GetRenderWindow("coronal");
  QmitkRenderWindow *mainWindow3d = m_ContainingFunctionality->GetRenderWindow("3d");

  // Main windows could be NULL if main window not initialised,
  // or no valid QmitkRenderer returned from mitkIRenderWindowPart.
  if (   mainWindowAxial == NULL
      || mainWindowSagittal == NULL
      || mainWindowCoronal == NULL
      || mainWindow3d == NULL
      )
  {
    return;
  }

  // Check if the user selected a completely different main window widget, or
  // if the user selected a different view (axial, coronal, sagittal) within
  // the same QmitkMIDASStdMultiWidget.
  bool mainWindowChanged = false;
  if (   mainWindowAxial    != m_MainWindowAxial
      || mainWindowSagittal != m_MainWindowSagittal
      || mainWindowCoronal  != m_MainWindowCoronal
      || mainWindow3d       != m_MainWindow3d
      )
  {
    mainWindowChanged = true;
  }

  // This will only be valid if we are not currently focused on THIS widget.
  // This should always be true at this point due to early exit above.
  MIDASView mainWindowView = this->GetCurrentMainWindowView();

  bool mainWindowViewChanged = false;
  if (mainWindowView != MIDAS_VIEW_UNKNOWN && mainWindowView != m_MainWindowView)
  {
    mainWindowViewChanged = true;
    m_MainWindowView = mainWindowView;
  }

  mitk::BaseRenderer* currentlyFocusedRenderer = this->GetCurrentlyFocusedRenderer();

  if (mainWindowChanged || m_CurrentRenderer == NULL || (mainWindowView != MIDAS_VIEW_UNKNOWN && m_View == MIDAS_VIEW_UNKNOWN))
  {
    mitk::SliceNavigationController::Pointer snc = mainWindowAxial->GetSliceNavigationController();
    assert(snc);

    mitk::Geometry3D::ConstPointer worldGeom = snc->GetInputWorldGeometry();
    if (worldGeom.IsNotNull())
    {
      mitk::Geometry3D::Pointer geom = const_cast<mitk::Geometry3D*>(worldGeom.GetPointer());
      assert(geom);

      m_MainWindowAxial = mainWindowAxial;
      m_MainWindowSagittal = mainWindowSagittal;
      m_MainWindowCoronal = mainWindowCoronal;
      m_MainWindow3d = mainWindow3d;

      m_ViewerWidget->SetGeometry(geom);
      m_ViewerWidget->SetBoundGeometryActive(false);
      m_ViewerWidget->SetNavigationControllerEventListening(true);
      m_ViewerWidget->SetDisplay2DCursorsLocally(true);
      m_ViewerWidget->SetShow3DWindowInOrthoView(true);
      if (!m_ViewerWidget->IsEnabled())
      {
        m_ViewerWidget->SetEnabled(true);
      }

      std::vector<mitk::DataNode*> crossHairs = m_ViewerWidget->GetWidgetPlanes();
      std::vector<mitk::BaseRenderer*> windowsToTrack;
      windowsToTrack.push_back(mitk::BaseRenderer::GetInstance(mainWindowAxial->GetVtkRenderWindow()));

      m_VisibilityTracker->SetRenderersToTrack(windowsToTrack);
      m_VisibilityTracker->SetNodesToIgnore(crossHairs);
      m_VisibilityTracker->OnPropertyChanged(); // force update

      m_ViewerWidget->FitToDisplay();
      this->ChangeLayout(true);
    }
  }
  else if (mainWindowViewChanged)
  {
    this->ChangeLayout(false);
  }
  m_CurrentRenderer = currentlyFocusedRenderer;
  m_ViewerWidget->RequestUpdate();
}


//-----------------------------------------------------------------------------
mitk::BaseRenderer* QmitkMIDASSegmentationViewWidget::GetCurrentlyFocusedRenderer() const
{
  mitk::BaseRenderer* result = NULL;

  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  if (focusManager != NULL)
  {
    result = focusManager->GetFocused();
  }
  return result;
}


//-----------------------------------------------------------------------------
bool QmitkMIDASSegmentationViewWidget::IsCurrentlyFocusedWindowInThisWidget()
{
  bool result = false;

  mitk::BaseRenderer* focusedRenderer = this->GetCurrentlyFocusedRenderer();
  if (focusedRenderer != NULL)
  {
    vtkRenderWindow* focusedVtkRenderWindow = focusedRenderer->GetRenderWindow();

    std::vector<QmitkRenderWindow*> renderWindows = m_ViewerWidget->GetRenderWindows();
    for (unsigned int i = 0; i < renderWindows.size(); i++)
    {
      if (renderWindows[i]->GetVtkRenderWindow() == focusedVtkRenderWindow)
      {
        result = true;
      }
    }
  }
  return result;
}


//-----------------------------------------------------------------------------
MIDASOrientation QmitkMIDASSegmentationViewWidget::GetCurrentMainWindowOrientation()
{
  MIDASOrientation orientation = MIDAS_ORIENTATION_UNKNOWN;

  mitk::BaseRenderer* focusedRenderer = this->GetCurrentlyFocusedRenderer();
  if (focusedRenderer != NULL)
  {
    vtkRenderWindow* focusedWindowRenderWindow = focusedRenderer->GetRenderWindow();

    QmitkRenderWindow *mainWindowAxial = m_ContainingFunctionality->GetRenderWindow("axial");
    QmitkRenderWindow *mainWindowSagittal = m_ContainingFunctionality->GetRenderWindow("sagittal");
    QmitkRenderWindow *mainWindowCoronal = m_ContainingFunctionality->GetRenderWindow("coronal");

    if (focusedWindowRenderWindow != NULL
        && mainWindowAxial != NULL
        && mainWindowSagittal != NULL
        && mainWindowCoronal != NULL
        )
    {
      if (focusedWindowRenderWindow == mainWindowAxial->GetVtkRenderWindow())
      {
        orientation = MIDAS_ORIENTATION_AXIAL;
      }
      else
      if (focusedWindowRenderWindow == mainWindowSagittal->GetVtkRenderWindow())
      {
        orientation = MIDAS_ORIENTATION_SAGITTAL;
      }
      else
      if (focusedWindowRenderWindow == mainWindowCoronal->GetVtkRenderWindow())
      {
        orientation = MIDAS_ORIENTATION_CORONAL;
      }
    }
  }
  return orientation;
}


//-----------------------------------------------------------------------------
MIDASView QmitkMIDASSegmentationViewWidget::GetCurrentMainWindowView()
{
  MIDASView mainWindowView = MIDAS_VIEW_UNKNOWN;
  MIDASOrientation mainWindowOrientation = this->GetCurrentMainWindowOrientation();

  if (mainWindowOrientation == MIDAS_ORIENTATION_AXIAL)
  {
    mainWindowView = MIDAS_VIEW_AXIAL;
  }
  else if (mainWindowOrientation == MIDAS_ORIENTATION_SAGITTAL)
  {
    mainWindowView = MIDAS_VIEW_SAGITTAL;
  }
  else if (mainWindowOrientation == MIDAS_ORIENTATION_CORONAL)
  {
    mainWindowView = MIDAS_VIEW_CORONAL;
  }
  return mainWindowView;
}
