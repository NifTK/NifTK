/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date$
 Revision          : $Revision$
 Last modified by  : $Author$

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "QmitkMIDASSegmentationViewWidget.h"
#include "QmitkMIDASBaseSegmentationFunctionality.h"
#include "QmitkMIDASSingleViewWidgetListVisibilityManager.h"
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
  renderers.push_back(mitk::BaseRenderer::GetInstance(m_ViewerWidget->GetAxialWindow()->GetVtkRenderWindow()));
  renderers.push_back(mitk::BaseRenderer::GetInstance(m_ViewerWidget->GetSagittalWindow()->GetVtkRenderWindow()));
  renderers.push_back(mitk::BaseRenderer::GetInstance(m_ViewerWidget->GetCoronalWindow()->GetVtkRenderWindow()));

  m_NodeAddedSetter = mitk::MIDASNodeAddedVisibilitySetter::New();
  m_NodeAddedSetter->SetRenderers(renderers);
  m_NodeAddedSetter->SetVisibility(false);

  m_VisibilityTracker = mitk::DataStorageVisibilityTracker::New();
  m_VisibilityTracker->SetRenderersToUpdate(renderers);

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
  this->m_ViewerWidget->SetEnabled(false);
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
  this->m_AxialRadioButton->setEnabled(enabled);
  this->m_SagittalRadioButton->setEnabled(enabled);
  this->m_CoronalRadioButton->setEnabled(enabled);
  this->m_OrthoRadioButton->setEnabled(enabled);
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
  if (this->m_AxialRadioButton->isChecked())
  {
    this->ChangeLayout();
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASSegmentationViewWidget::OnCoronalToggled(bool)
{
  if (this->m_CoronalRadioButton->isChecked())
  {
    this->ChangeLayout();
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASSegmentationViewWidget::OnSagittalToggled(bool)
{
  if (this->m_SagittalRadioButton->isChecked())
  {
    this->ChangeLayout();
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASSegmentationViewWidget::OnOrthoToggled(bool)
{
  if (this->m_OrthoRadioButton->isChecked())
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
        nextView = MIDAS_VIEW_SAG_COR_V;
      }
      else if (m_MainWindowView == MIDAS_VIEW_SAGITTAL)
      {
        nextView = MIDAS_VIEW_AX_COR_V;
      }
      else if (m_MainWindowView == MIDAS_VIEW_CORONAL)
      {
        nextView = MIDAS_VIEW_AX_SAG_V;
      }
    }
    else
    {
      if (m_MainWindowView == MIDAS_VIEW_AXIAL)
      {
        nextView = MIDAS_VIEW_SAG_COR_H;
      }
      else if (m_MainWindowView == MIDAS_VIEW_SAGITTAL)
      {
        nextView = MIDAS_VIEW_AX_COR_H;
      }
      else if (m_MainWindowView == MIDAS_VIEW_CORONAL)
      {
        nextView = MIDAS_VIEW_AX_SAG_H;
      }
    }
  }
  else
  {
    if (m_MainWindowView == MIDAS_VIEW_AXIAL)
    {
      if (this->m_SagittalRadioButton->isChecked())
      {
        nextView = MIDAS_VIEW_SAGITTAL;
      }
      else if (this->m_OrthoRadioButton->isChecked())
      {
        nextView = MIDAS_VIEW_ORTHO;
      }
      else
      {
        nextView = MIDAS_VIEW_CORONAL;
        this->m_CoronalRadioButton->setChecked(true);
      }
    }
    else if (m_MainWindowView == MIDAS_VIEW_SAGITTAL)
    {
      if (this->m_AxialRadioButton->isChecked())
      {
        nextView = MIDAS_VIEW_AXIAL;
      }
      else if (this->m_OrthoRadioButton->isChecked())
      {
        nextView = MIDAS_VIEW_ORTHO;
      }
      else
      {
        nextView = MIDAS_VIEW_CORONAL;
        this->m_CoronalRadioButton->setChecked(true);
      }
    }
    else if (m_MainWindowView == MIDAS_VIEW_CORONAL)
    {
      if (this->m_SagittalRadioButton->isChecked())
      {
        nextView = MIDAS_VIEW_SAGITTAL;
      }
      else if (this->m_OrthoRadioButton->isChecked())
      {
        nextView = MIDAS_VIEW_ORTHO;
      }
      else
      {
        nextView = MIDAS_VIEW_AXIAL;
        this->m_AxialRadioButton->setChecked(true);
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
      this->m_ViewerWidget->SetView(this->m_View, false);
    }
    else
    {
      this->m_ViewerWidget->SwitchView(this->m_View);
    }
    emit ViewChanged(this->m_View);
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
      this->m_AxialRadioButton->setEnabled(false);
    }
    else if (m_MainWindowView == MIDAS_VIEW_SAGITTAL)
    {
      this->m_SagittalRadioButton->setEnabled(false);
    }
    else if (m_MainWindowView == MIDAS_VIEW_CORONAL)
    {
      this->m_CoronalRadioButton->setEnabled(false);
    }
    else if (m_MainWindowView == MIDAS_VIEW_ORTHO)
    {
      this->m_OrthoRadioButton->setEnabled(false);
    }
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASSegmentationViewWidget::OnFocusChanged()
{
  // If the newly focussed window is this widget, nothing to update. Stop early.
  if (this->IsCurrentlyFocussedWindowInThisWidget())
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

  // This will only be valid if we are not currently focussed on THIS widget.
  // This should always be true at this point due to early exit above.
  MIDASView mainWindowView = this->GetCurrentMainWindowView();

  bool mainWindowViewChanged = false;
  if (mainWindowView != MIDAS_VIEW_UNKNOWN && mainWindowView != this->m_MainWindowView)
  {
    mainWindowViewChanged = true;
    this->m_MainWindowView = mainWindowView;
  }

  mitk::BaseRenderer* currentlyFocussedRenderer = this->GetCurrentlyFocussedRenderer();

  if (mainWindowChanged || m_CurrentRenderer == NULL || (mainWindowView != MIDAS_VIEW_UNKNOWN && this->m_View == MIDAS_VIEW_UNKNOWN))
  {
    mitk::SliceNavigationController::Pointer snc = mainWindowAxial->GetSliceNavigationController();
    assert(snc);

    mitk::Geometry3D::ConstPointer worldGeom = snc->GetInputWorldGeometry();
    if (worldGeom.IsNotNull())
    {
      mitk::Geometry3D::Pointer geom = const_cast<mitk::Geometry3D*>(worldGeom.GetPointer());
      assert(geom);

      this->m_MainWindowAxial = mainWindowAxial;
      this->m_MainWindowSagittal = mainWindowSagittal;
      this->m_MainWindowCoronal = mainWindowCoronal;
      this->m_MainWindow3d = mainWindow3d;

      this->m_ViewerWidget->SetGeometry(geom);
      this->m_ViewerWidget->SetBoundGeometryActive(false);
      this->m_ViewerWidget->SetNavigationControllerEventListening(true);
      this->m_ViewerWidget->SetDisplay2DCursorsLocally(true);
      this->m_ViewerWidget->SetDisplay3DViewInOrthoView(true);
      if (!this->m_ViewerWidget->IsEnabled())
      {
        this->m_ViewerWidget->SetEnabled(true);
      }

      std::vector<mitk::DataNode*> crossHairs = m_ViewerWidget->GetWidgetPlanes();
      std::vector<mitk::BaseRenderer*> windowsToTrack;
      windowsToTrack.push_back(mitk::BaseRenderer::GetInstance(mainWindowAxial->GetVtkRenderWindow()));

      this->m_VisibilityTracker->SetRenderersToTrack(windowsToTrack);
      this->m_VisibilityTracker->SetNodesToIgnore(crossHairs);
      this->m_VisibilityTracker->OnPropertyChanged(); // force update

      this->m_ViewerWidget->FitToDisplay();
      this->ChangeLayout(true);
    }
  }
  else if (mainWindowViewChanged)
  {
    this->ChangeLayout(false);
  }
  this->m_CurrentRenderer = currentlyFocussedRenderer;
  this->m_ViewerWidget->RequestUpdate();
}


//-----------------------------------------------------------------------------
mitk::BaseRenderer* QmitkMIDASSegmentationViewWidget::GetCurrentlyFocussedRenderer() const
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
bool QmitkMIDASSegmentationViewWidget::IsCurrentlyFocussedWindowInThisWidget()
{
  bool result = false;

  mitk::BaseRenderer* focussedRenderer = this->GetCurrentlyFocussedRenderer();
  if (focussedRenderer != NULL)
  {
    vtkRenderWindow* focusedWindowRenderWindow = focussedRenderer->GetRenderWindow();

    std::vector<vtkRenderWindow*> windowsInThisWidget = m_ViewerWidget->GetAllVtkWindows();
    for (unsigned int i = 0; i < windowsInThisWidget.size(); i++)
    {
      if (windowsInThisWidget[i] == focusedWindowRenderWindow)
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

  mitk::BaseRenderer* focussedRenderer = this->GetCurrentlyFocussedRenderer();
  if (focussedRenderer != NULL)
  {
    vtkRenderWindow* focusedWindowRenderWindow = focussedRenderer->GetRenderWindow();

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
