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
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QSpacerItem>
#include <mitkDataStorage.h>
#include <mitkGlobalInteraction.h>
#include <mitkFocusManager.h>
#include <mitkGeometry3D.h>
#include <mitkSliceNavigationController.h>
#include <mitkBaseRenderer.h>
#include <itkCommand.h>


//-----------------------------------------------------------------------------
QmitkMIDASSegmentationViewWidget::QmitkMIDASSegmentationViewWidget(QWidget* parent)
: m_ContainingFunctionality(NULL)
, m_FocusManagerObserverTag(0)
, m_WindowLayout(WINDOW_LAYOUT_UNKNOWN)
, m_MainWindowLayout(WINDOW_LAYOUT_UNKNOWN)
, m_MainAxialWindow(NULL)
, m_MainSagittalWindow(NULL)
, m_MainCoronalWindow(NULL)
, m_Main3DWindow(NULL)
, m_CurrentRenderer(NULL)
, m_NodeAddedSetter(NULL)
, m_VisibilityTracker(NULL)
, m_Magnification(0.0)
, m_SingleWindowLayouts()
{
  this->setupUi(parent);

  m_ViewerWidget->SetSelected(false);

  m_MultiWindowComboBox->addItem("2H");
  m_MultiWindowComboBox->addItem("2V");
  m_MultiWindowComboBox->addItem("2x2");

  m_CoronalWindowRadioButton->setChecked(true);

  m_SingleWindowLayouts[WINDOW_LAYOUT_AXIAL] = WINDOW_LAYOUT_CORONAL;
  m_SingleWindowLayouts[WINDOW_LAYOUT_SAGITTAL] = WINDOW_LAYOUT_CORONAL;
  m_SingleWindowLayouts[WINDOW_LAYOUT_CORONAL] = WINDOW_LAYOUT_SAGITTAL;

  this->ChangeLayout();

  m_MagnificationSpinBox->setDecimals(2);
  m_MagnificationSpinBox->setSingleStep(1.0);

  double minMagnification = std::ceil(m_ViewerWidget->GetMinMagnification());
  double maxMagnification = std::floor(m_ViewerWidget->GetMaxMagnification());

  m_MagnificationSpinBox->setMinimum(minMagnification);
  m_MagnificationSpinBox->setMaximum(maxMagnification);

  m_ControlsWidget->setEnabled(false);

  std::vector<mitk::BaseRenderer*> renderers;
  renderers.push_back(m_ViewerWidget->GetAxialWindow()->GetRenderer());
  renderers.push_back(m_ViewerWidget->GetSagittalWindow()->GetRenderer());
  renderers.push_back(m_ViewerWidget->GetCoronalWindow()->GetRenderer());

  m_NodeAddedSetter = mitk::DataNodeAddedVisibilitySetter::New();
  m_MIDASToolNodeNameFilter = mitk::MIDASDataNodeNameStringFilter::New();
  m_NodeAddedSetter->AddFilter(m_MIDASToolNodeNameFilter.GetPointer());
  m_NodeAddedSetter->SetRenderers(renderers);
  m_NodeAddedSetter->SetVisibility(false);

  m_VisibilityTracker = mitk::DataStorageVisibilityTracker::New();
  m_VisibilityTracker->SetNodesToIgnore(m_ViewerWidget->GetWidgetPlanes());
  m_VisibilityTracker->SetRenderersToUpdate(renderers);

  m_ViewerWidget->SetDisplay2DCursorsGlobally(false);
  m_ViewerWidget->SetDisplay2DCursorsLocally(true);
  m_ViewerWidget->SetRememberSettingsPerLayout(true);
  m_ViewerWidget->SetDisplayInteractionsEnabled(true);
  m_ViewerWidget->SetCursorPositionsBound(false);
  m_ViewerWidget->SetScaleFactorsBound(true);

  connect(m_AxialWindowRadioButton, SIGNAL(toggled(bool)), this, SLOT(OnAxialWindowRadioButtonToggled(bool)));
  connect(m_SagittalWindowRadioButton, SIGNAL(toggled(bool)), this, SLOT(OnSagittalWindowRadioButtonToggled(bool)));
  connect(m_CoronalWindowRadioButton, SIGNAL(toggled(bool)), this, SLOT(OnCoronalWindowRadioButtonToggled(bool)));
  connect(m_MultiWindowRadioButton, SIGNAL(toggled(bool)), this, SLOT(OnMultiWindowRadioButtonToggled(bool)));
  connect(m_MultiWindowComboBox, SIGNAL(currentIndexChanged(int)), SLOT(OnMultiWindowComboBoxIndexChanged()));

  connect(m_MagnificationSpinBox, SIGNAL(valueChanged(double)), this, SLOT(OnMagnificationChanged(double)));
  connect(m_ViewerWidget, SIGNAL(ScaleFactorChanged(niftkSingleViewerWidget*, double)), this, SLOT(OnScaleFactorChanged(niftkSingleViewerWidget*, double)));
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
    m_ViewerWidget->SetSelected(false);
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
void QmitkMIDASSegmentationViewWidget::SetEnabled(bool enabled)
{
  m_ControlsWidget->setEnabled(enabled);
}


//-----------------------------------------------------------------------------
void QmitkMIDASSegmentationViewWidget::OnAxialWindowRadioButtonToggled(bool checked)
{
  if (checked)
  {
    WindowLayout mainWindowLayout = this->GetCurrentMainWindowLayout();

    if (::IsSingleWindowLayout(mainWindowLayout))
    {
      m_SingleWindowLayouts[mainWindowLayout] = WINDOW_LAYOUT_AXIAL;
    }
    this->ChangeLayout();
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASSegmentationViewWidget::OnSagittalWindowRadioButtonToggled(bool checked)
{
  if (checked)
  {
    WindowLayout mainWindowLayout = this->GetCurrentMainWindowLayout();

    if (::IsSingleWindowLayout(mainWindowLayout))
    {
      m_SingleWindowLayouts[mainWindowLayout] = WINDOW_LAYOUT_SAGITTAL;
    }
    this->ChangeLayout();
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASSegmentationViewWidget::OnCoronalWindowRadioButtonToggled(bool checked)
{
  if (checked)
  {
    WindowLayout mainWindowLayout = this->GetCurrentMainWindowLayout();

    if (::IsSingleWindowLayout(mainWindowLayout))
    {
      m_SingleWindowLayouts[mainWindowLayout] = WINDOW_LAYOUT_CORONAL;
    }
    this->ChangeLayout();
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASSegmentationViewWidget::OnMultiWindowRadioButtonToggled(bool checked)
{
  if (checked)
  {
    this->ChangeLayout();
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASSegmentationViewWidget::OnMultiWindowComboBoxIndexChanged()
{
  m_MultiWindowRadioButton->setChecked(true);
  this->ChangeLayout();
}

//-----------------------------------------------------------------------------
void QmitkMIDASSegmentationViewWidget::ChangeLayout()
{
  WindowLayout nextLayout = WINDOW_LAYOUT_UNKNOWN;

  bool wasBlocked = m_LayoutWidget->blockSignals(true);

  if (m_MultiWindowRadioButton->isChecked())
  {
    // 2H
    if (m_MultiWindowComboBox->currentIndex() == 0)
    {
      if (m_MainWindowLayout == WINDOW_LAYOUT_AXIAL)
      {
        nextLayout = WINDOW_LAYOUT_COR_SAG_H;
      }
      else if (m_MainWindowLayout == WINDOW_LAYOUT_SAGITTAL)
      {
        nextLayout = WINDOW_LAYOUT_COR_AX_H;
      }
      else if (m_MainWindowLayout == WINDOW_LAYOUT_CORONAL)
      {
        nextLayout = WINDOW_LAYOUT_SAG_AX_H;
      }
    }
    // 2V
    else if (m_MultiWindowComboBox->currentIndex() == 1)
    {
      if (m_MainWindowLayout == WINDOW_LAYOUT_AXIAL)
      {
        nextLayout = WINDOW_LAYOUT_COR_SAG_V;
      }
      else if (m_MainWindowLayout == WINDOW_LAYOUT_SAGITTAL)
      {
        nextLayout = WINDOW_LAYOUT_COR_AX_V;
      }
      else if (m_MainWindowLayout == WINDOW_LAYOUT_CORONAL)
      {
        nextLayout = WINDOW_LAYOUT_SAG_AX_V;
      }
    }
    // 2x2
    else if (m_MultiWindowComboBox->currentIndex() == 2)
    {
      nextLayout = WINDOW_LAYOUT_ORTHO;
    }
  }
  else if (::IsSingleWindowLayout(m_MainWindowLayout) && m_MainWindowLayout != WINDOW_LAYOUT_3D)
  {
    nextLayout = m_SingleWindowLayouts[m_MainWindowLayout];

    QRadioButton* nextLayoutRadioButton = 0;
    if (nextLayout == WINDOW_LAYOUT_AXIAL && !m_AxialWindowRadioButton->isChecked())
    {
      nextLayoutRadioButton = m_AxialWindowRadioButton;
    }
    else if (nextLayout == WINDOW_LAYOUT_SAGITTAL && !m_SagittalWindowRadioButton->isChecked())
    {
      nextLayoutRadioButton = m_SagittalWindowRadioButton;
    }
    if (nextLayout == WINDOW_LAYOUT_CORONAL && !m_CoronalWindowRadioButton->isChecked())
    {
      nextLayoutRadioButton = m_CoronalWindowRadioButton;
    }

    if (nextLayoutRadioButton)
    {
      nextLayoutRadioButton->setChecked(true);
    }
  }

  if (!m_MultiWindowRadioButton->isChecked())
  {
    m_ControlsWidget->setEnabled(true);
    m_AxialWindowRadioButton->setEnabled(m_MainWindowLayout != WINDOW_LAYOUT_AXIAL);
    m_SagittalWindowRadioButton->setEnabled(m_MainWindowLayout != WINDOW_LAYOUT_SAGITTAL);
    m_CoronalWindowRadioButton->setEnabled(m_MainWindowLayout != WINDOW_LAYOUT_CORONAL);
  }

  m_LayoutWidget->blockSignals(wasBlocked);

  if (nextLayout != WINDOW_LAYOUT_UNKNOWN && nextLayout != m_WindowLayout)
  {
    m_WindowLayout = nextLayout;
    m_ViewerWidget->SetLayout(m_WindowLayout);

    double magnification = m_ViewerWidget->GetMagnification();

    bool wasBlocked = m_MagnificationSpinBox->blockSignals(true);
    m_MagnificationSpinBox->setValue(magnification);
    m_MagnificationSpinBox->blockSignals(wasBlocked);

    emit LayoutChanged(m_WindowLayout);
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASSegmentationViewWidget::OnFocusChanged()
{
  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  mitk::BaseRenderer* focusedRenderer = focusManager->GetFocused();

  QmitkRenderWindow* renderWindow = m_ViewerWidget->GetRenderWindow(focusedRenderer->GetRenderWindow());

  // If the newly focused window is this widget, nothing to update. Stop early.
  if (renderWindow)
  {
    m_ViewerWidget->SetSelectedRenderWindow(renderWindow);

    double magnification = m_ViewerWidget->GetMagnification();
    m_Magnification = magnification;

    bool wasBlocked = m_MagnificationSpinBox->blockSignals(true);
    m_MagnificationSpinBox->setValue(magnification);
    m_MagnificationSpinBox->blockSignals(wasBlocked);

    return;
  }

  // Get hold of main windows, using QmitkAbstractView lookup mitkIRenderWindowPart.
  QmitkRenderWindow* mainAxialWindow = m_ContainingFunctionality->GetRenderWindow("axial");
  QmitkRenderWindow* mainSagittalWindow = m_ContainingFunctionality->GetRenderWindow("sagittal");
  QmitkRenderWindow* mainCoronalWindow = m_ContainingFunctionality->GetRenderWindow("coronal");
  QmitkRenderWindow* main3DWindow = m_ContainingFunctionality->GetRenderWindow("3d");

  // Main windows could be NULL if main window not initialised,
  // or no valid QmitkRenderer returned from mitkIRenderWindowPart.
  if (   mainAxialWindow == NULL
      || mainSagittalWindow == NULL
      || mainCoronalWindow == NULL
      || main3DWindow == NULL
      )
  {
    return;
  }

  // Check if the user selected a completely different main window widget, or
  // if the user selected a different layout (axial, coronal, sagittal) within
  // the same DnDMultiWindowWidget.
  bool mainWindowChanged = false;
  if (   mainAxialWindow    != m_MainAxialWindow
      || mainSagittalWindow != m_MainSagittalWindow
      || mainCoronalWindow  != m_MainCoronalWindow
      || main3DWindow       != m_Main3DWindow
      )
  {
    mainWindowChanged = true;
  }

  if (mainWindowChanged)
  {
    QmitkRenderWindow* axialWindow = m_ViewerWidget->GetAxialWindow();
    if (mainAxialWindow != NULL)
    {
      axialWindow->GetSliceNavigationController()->ConnectGeometryEvents(mainAxialWindow->GetSliceNavigationController());
      mainAxialWindow->GetSliceNavigationController()->ConnectGeometryEvents(axialWindow->GetSliceNavigationController());
    }
    else
    {
      axialWindow->GetSliceNavigationController()->Disconnect(m_MainAxialWindow->GetSliceNavigationController());
      m_MainAxialWindow->GetSliceNavigationController()->Disconnect(axialWindow->GetSliceNavigationController());
    }
    QmitkRenderWindow* sagittalWindow = m_ViewerWidget->GetSagittalWindow();
    if (mainSagittalWindow != NULL)
    {
      sagittalWindow->GetSliceNavigationController()->ConnectGeometryEvents(mainSagittalWindow->GetSliceNavigationController());
      mainSagittalWindow->GetSliceNavigationController()->ConnectGeometryEvents(sagittalWindow->GetSliceNavigationController());
    }
    else
    {
      sagittalWindow->GetSliceNavigationController()->Disconnect(m_MainSagittalWindow->GetSliceNavigationController());
      m_MainSagittalWindow->GetSliceNavigationController()->Disconnect(sagittalWindow->GetSliceNavigationController());
    }
    QmitkRenderWindow* coronalWindow = m_ViewerWidget->GetCoronalWindow();
    if (mainCoronalWindow != NULL)
    {
      coronalWindow->GetSliceNavigationController()->ConnectGeometryEvents(mainCoronalWindow->GetSliceNavigationController());
      mainCoronalWindow->GetSliceNavigationController()->ConnectGeometryEvents(coronalWindow->GetSliceNavigationController());
    }
    else
    {
      coronalWindow->GetSliceNavigationController()->Disconnect(m_MainCoronalWindow->GetSliceNavigationController());
      m_MainCoronalWindow->GetSliceNavigationController()->Disconnect(coronalWindow->GetSliceNavigationController());
    }
  }

  // This will only be valid if we are not currently focused on THIS widget.
  // This should always be true at this point due to early exit above.
  WindowLayout mainWindowLayout = this->GetCurrentMainWindowLayout();

  bool mainWindowLayoutChanged = false;
  if (mainWindowLayout != WINDOW_LAYOUT_UNKNOWN && mainWindowLayout != m_MainWindowLayout)
  {
    mainWindowLayoutChanged = true;
    m_MainWindowLayout = mainWindowLayout;
  }

  if (mainWindowChanged || m_CurrentRenderer == NULL || (mainWindowLayout != WINDOW_LAYOUT_UNKNOWN && m_WindowLayout == WINDOW_LAYOUT_UNKNOWN))
  {
    const mitk::TimeGeometry* worldTimeGeometry = mainAxialWindow->GetRenderer()->GetTimeWorldGeometry();
    if (worldTimeGeometry)
    {
      mitk::TimeGeometry::Pointer timeGeometry = const_cast<mitk::TimeGeometry*>(worldTimeGeometry);
      assert(timeGeometry);

      m_MainAxialWindow = mainAxialWindow;
      m_MainSagittalWindow = mainSagittalWindow;
      m_MainCoronalWindow = mainCoronalWindow;
      m_Main3DWindow = main3DWindow;

      m_ViewerWidget->SetGeometry(timeGeometry);
      m_ViewerWidget->SetBoundGeometryActive(false);
      m_ViewerWidget->SetShow3DWindowInOrthoView(true);
      if (!m_ViewerWidget->IsEnabled())
      {
        m_ViewerWidget->SetEnabled(true);
      }

      std::vector<mitk::DataNode*> crossHairs = m_ViewerWidget->GetWidgetPlanes();
      std::vector<mitk::BaseRenderer*> renderersToTrack;
      renderersToTrack.push_back(mainAxialWindow->GetRenderer());
      renderersToTrack.push_back(mainSagittalWindow->GetRenderer());
      renderersToTrack.push_back(mainCoronalWindow->GetRenderer());

      m_VisibilityTracker->SetRenderersToTrack(renderersToTrack);
      m_VisibilityTracker->SetNodesToIgnore(crossHairs);
      m_VisibilityTracker->OnPropertyChanged(); // force update

      m_ViewerWidget->FitToDisplay();
      this->ChangeLayout();
    }
  }
  else if (mainWindowLayoutChanged)
  {
    this->ChangeLayout();
  }

  m_CurrentRenderer = focusedRenderer;

  m_ViewerWidget->RequestUpdate();
}


//-----------------------------------------------------------------------------
MIDASOrientation QmitkMIDASSegmentationViewWidget::GetCurrentMainWindowOrientation()
{
  MIDASOrientation orientation = MIDAS_ORIENTATION_UNKNOWN;

  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  mitk::BaseRenderer* focusedRenderer = focusManager->GetFocused();

  if (focusedRenderer != NULL)
  {
    QmitkRenderWindow* mainWindowAxial = m_ContainingFunctionality->GetRenderWindow("axial");
    QmitkRenderWindow* mainWindowSagittal = m_ContainingFunctionality->GetRenderWindow("sagittal");
    QmitkRenderWindow* mainWindowCoronal = m_ContainingFunctionality->GetRenderWindow("coronal");

    if (mainWindowAxial && mainWindowSagittal && mainWindowCoronal)
    {
      if (focusedRenderer == mainWindowAxial->GetRenderer())
      {
        orientation = MIDAS_ORIENTATION_AXIAL;
      }
      else if (focusedRenderer == mainWindowSagittal->GetRenderer())
      {
        orientation = MIDAS_ORIENTATION_SAGITTAL;
      }
      else if (focusedRenderer == mainWindowCoronal->GetRenderer())
      {
        orientation = MIDAS_ORIENTATION_CORONAL;
      }
    }
  }
  return orientation;
}


//-----------------------------------------------------------------------------
WindowLayout QmitkMIDASSegmentationViewWidget::GetCurrentMainWindowLayout()
{
  WindowLayout mainWindowLayout = WINDOW_LAYOUT_UNKNOWN;
  MIDASOrientation mainWindowOrientation = this->GetCurrentMainWindowOrientation();

  if (mainWindowOrientation == MIDAS_ORIENTATION_AXIAL)
  {
    mainWindowLayout = WINDOW_LAYOUT_AXIAL;
  }
  else if (mainWindowOrientation == MIDAS_ORIENTATION_SAGITTAL)
  {
    mainWindowLayout = WINDOW_LAYOUT_SAGITTAL;
  }
  else if (mainWindowOrientation == MIDAS_ORIENTATION_CORONAL)
  {
    mainWindowLayout = WINDOW_LAYOUT_CORONAL;
  }
  return mainWindowLayout;
}


//-----------------------------------------------------------------------------
void QmitkMIDASSegmentationViewWidget::OnScaleFactorChanged(niftkSingleViewerWidget*, double scaleFactor)
{
  double magnification = m_ViewerWidget->GetMagnification();

  bool wasBlocked = m_MagnificationSpinBox->blockSignals(true);
  m_MagnificationSpinBox->setValue(magnification);
  m_MagnificationSpinBox->blockSignals(wasBlocked);

  m_Magnification = magnification;
}


//-----------------------------------------------------------------------------
void QmitkMIDASSegmentationViewWidget::OnMagnificationChanged(double magnification)
{
  double roundedMagnification = std::floor(magnification);

  // If we are between two integers, we raise a new event:
  if (magnification != roundedMagnification)
  {
    double newMagnification = roundedMagnification;
    // If the value has decreased, we have to increase the rounded value.
    if (magnification < m_Magnification)
    {
      newMagnification += 1.0;
    }

    m_MagnificationSpinBox->setValue(newMagnification);
  }
  else
  {
    m_ViewerWidget->SetMagnification(magnification);
    m_Magnification = magnification;
  }
}
