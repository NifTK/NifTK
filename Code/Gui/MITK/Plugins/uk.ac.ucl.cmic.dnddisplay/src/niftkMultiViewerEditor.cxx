/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkMultiViewerEditor.h"

#include <berryUIException.h>
#include <berryIWorkbenchPage.h>
#include <berryIPreferencesService.h>

#include <QWidget>
#include <QGridLayout>

#include <mitkGlobalInteraction.h>
#include <mitkIDataStorageService.h>
#include <mitkNodePredicateNot.h>
#include <mitkNodePredicateProperty.h>

#include <niftkMultiViewerWidget.h>
#include <niftkMultiViewerVisibilityManager.h>
#include "niftkMultiViewerEditorPreferencePage.h"

const std::string niftkMultiViewerEditor::EDITOR_ID = "org.mitk.editors.dnddisplay";

class niftkMultiViewerEditorPrivate
{
public:
  niftkMultiViewerEditorPrivate();
  ~niftkMultiViewerEditorPrivate();

  niftkMultiViewerWidget* m_MultiViewer;
  niftkMultiViewerVisibilityManager* m_MultiViewerVisibilityManager;
  mitk::RenderingManager::Pointer m_RenderingManager;
  berry::IPartListener::Pointer m_PartListener;
  mitk::IRenderingManager* m_RenderingManagerInterface;
};

//-----------------------------------------------------------------------------
struct niftkMultiViewerEditorPartListener : public berry::IPartListener
{
  berryObjectMacro(niftkMultiViewerEditorPartListener)

  //-----------------------------------------------------------------------------
  niftkMultiViewerEditorPartListener(niftkMultiViewerEditorPrivate* dd)
    : d(dd)
  {}


  //-----------------------------------------------------------------------------
  Events::Types GetPartEventTypes() const
  {
    return Events::CLOSED | Events::HIDDEN | Events::VISIBLE;
  }


  //-----------------------------------------------------------------------------
  void PartClosed (berry::IWorkbenchPartReference::Pointer partRef)
  {
    if (partRef->GetId() == niftkMultiViewerEditor::EDITOR_ID)
    {
      niftkMultiViewerEditor::Pointer multiViewerEditor = partRef->GetPart(false).Cast<niftkMultiViewerEditor>();

      if (multiViewerEditor.IsNotNull()
        && multiViewerEditor->GetMultiViewer() == d->m_MultiViewer)
      {
        d->m_MultiViewer->Deactivated();
      }
    }
  }


  //-----------------------------------------------------------------------------
  void PartHidden (berry::IWorkbenchPartReference::Pointer partRef)
  {
    if (partRef->GetId() == niftkMultiViewerEditor::EDITOR_ID)
    {
      niftkMultiViewerEditor::Pointer multiViewerEditor = partRef->GetPart(false).Cast<niftkMultiViewerEditor>();

      if (multiViewerEditor.IsNotNull()
        && multiViewerEditor->GetMultiViewer() == d->m_MultiViewer)
      {
        d->m_MultiViewer->Deactivated();
      }
    }
  }


  //-----------------------------------------------------------------------------
  void PartVisible (berry::IWorkbenchPartReference::Pointer partRef)
  {
    if (partRef->GetId() == niftkMultiViewerEditor::EDITOR_ID)
    {
      niftkMultiViewerEditor::Pointer multiViewerEditor = partRef->GetPart(false).Cast<niftkMultiViewerEditor>();

      if (multiViewerEditor.IsNotNull()
        && multiViewerEditor->GetMultiViewer() == d->m_MultiViewer)
      {
        d->m_MultiViewer->Activated();
      }
    }
  }

private:

  niftkMultiViewerEditorPrivate* const d;

};


//-----------------------------------------------------------------------------
niftkMultiViewerEditorPrivate::niftkMultiViewerEditorPrivate()
: m_MultiViewer(0)
, m_MultiViewerVisibilityManager(0)
, m_RenderingManager(0)
, m_PartListener(new niftkMultiViewerEditorPartListener(this))
, m_RenderingManagerInterface(0)
{
  m_RenderingManager = mitk::RenderingManager::GetInstance();
  m_RenderingManager->SetConstrainedPaddingZooming(false);
  m_RenderingManagerInterface = mitk::MakeRenderingManagerInterface(m_RenderingManager);
}


//-----------------------------------------------------------------------------
niftkMultiViewerEditorPrivate::~niftkMultiViewerEditorPrivate()
{
  if (m_MultiViewerVisibilityManager != NULL)
  {
    delete m_MultiViewerVisibilityManager;
  }

  if (m_RenderingManagerInterface != NULL)
  {
    delete m_RenderingManagerInterface;
  }
}


//-----------------------------------------------------------------------------
niftkMultiViewerEditor::niftkMultiViewerEditor()
: d(new niftkMultiViewerEditorPrivate)
{
}


//-----------------------------------------------------------------------------
niftkMultiViewerEditor::~niftkMultiViewerEditor()
{
  this->GetSite()->GetPage()->RemovePartListener(d->m_PartListener);
}


//-----------------------------------------------------------------------------
void niftkMultiViewerEditor::CreateQtPartControl(QWidget* parent)
{
  if (d->m_MultiViewer == NULL)
  {
    mitk::DataStorage::Pointer dataStorage = this->GetDataStorage();
    assert(dataStorage);

    berry::IPreferencesService::Pointer prefService = berry::Platform::GetServiceRegistry().GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);
    berry::IBerryPreferences::Pointer prefs = (prefService->GetSystemPreferences()->Node(EDITOR_ID)).Cast<berry::IBerryPreferences>();
    assert( prefs );

    DnDDisplayInterpolationType defaultInterpolationType =
        (DnDDisplayInterpolationType)(prefs->GetInt(niftkMultiViewerEditorPreferencePage::DEFAULT_INTERPOLATION_TYPE, 2));
    WindowLayout defaultLayout =
        (WindowLayout)(prefs->GetInt(niftkMultiViewerEditorPreferencePage::MIDAS_DEFAULT_WINDOW_LAYOUT, 2)); // default = coronal
    DnDDisplayDropType defaultDropType =
        (DnDDisplayDropType)(prefs->GetInt(niftkMultiViewerEditorPreferencePage::DEFAULT_DROP_TYPE, 0));

    int defaultNumberOfRows = prefs->GetInt(niftkMultiViewerEditorPreferencePage::MIDAS_DEFAULT_VIEW_ROW_NUMBER, 1);
    int defaultNumberOfColumns = prefs->GetInt(niftkMultiViewerEditorPreferencePage::MIDAS_DEFAULT_VIEW_COLUMN_NUMBER, 1);
    bool showDropTypeControls = prefs->GetBool(niftkMultiViewerEditorPreferencePage::MIDAS_SHOW_DROP_TYPE_CONTROLS, false);
    bool showDirectionAnnotations = prefs->GetBool(niftkMultiViewerEditorPreferencePage::MIDAS_SHOW_DIRECTION_ANNOTATIONS, true);
    bool showShowingOptions = prefs->GetBool(niftkMultiViewerEditorPreferencePage::MIDAS_SHOW_SHOWING_OPTIONS, true);
    bool showWindowLayoutControls = prefs->GetBool(niftkMultiViewerEditorPreferencePage::MIDAS_SHOW_WINDOW_LAYOUT_CONTROLS, true);
    bool showViewerNumberControls = prefs->GetBool(niftkMultiViewerEditorPreferencePage::MIDAS_SHOW_VIEW_NUMBER_CONTROLS, true);
    bool showMagnificationSlider = prefs->GetBool(niftkMultiViewerEditorPreferencePage::MIDAS_SHOW_MAGNIFICATION_SLIDER, true);
    bool show3DWindowInMultiWindowLayout = prefs->GetBool(niftkMultiViewerEditorPreferencePage::MIDAS_SHOW_3D_WINDOW_IN_MULTI_WINDOW_LAYOUT, false);
    bool show2DCursors = prefs->GetBool(niftkMultiViewerEditorPreferencePage::MIDAS_SHOW_2D_CURSORS, true);
    bool rememberSettingsPerLayout = prefs->GetBool(niftkMultiViewerEditorPreferencePage::MIDAS_REMEMBER_VIEW_SETTINGS_PER_WINDOW_LAYOUT, true);
    bool sliceIndexTracking = prefs->GetBool(niftkMultiViewerEditorPreferencePage::MIDAS_SLICE_SELECT_TRACKING, true);
    bool magnificationTracking = prefs->GetBool(niftkMultiViewerEditorPreferencePage::MIDAS_MAGNIFICATION_SELECT_TRACKING, true);
    bool timeStepTracking = prefs->GetBool(niftkMultiViewerEditorPreferencePage::MIDAS_TIME_SELECT_TRACKING, true);

    d->m_MultiViewerVisibilityManager = new niftkMultiViewerVisibilityManager(dataStorage);
    d->m_MultiViewerVisibilityManager->SetInterpolationType(defaultInterpolationType);
    d->m_MultiViewerVisibilityManager->SetDefaultWindowLayout(defaultLayout);
    d->m_MultiViewerVisibilityManager->SetDropType(defaultDropType);

    d->m_RenderingManager->SetDataStorage(dataStorage);

    // Create the niftkMultiViewerWidget
    d->m_MultiViewer = new niftkMultiViewerWidget(
        d->m_MultiViewerVisibilityManager,
        d->m_RenderingManager,
        dataStorage,
        defaultNumberOfRows,
        defaultNumberOfColumns,
        parent);

    // Setup GUI a bit more.
    d->m_MultiViewer->SetDropType(defaultDropType);
    d->m_MultiViewer->SetShowOptionsVisible(showShowingOptions);
    d->m_MultiViewer->SetWindowLayoutControlsVisible(showWindowLayoutControls);
    d->m_MultiViewer->SetViewerNumberControlsVisible(showViewerNumberControls);
    d->m_MultiViewer->SetShowDropTypeControls(showDropTypeControls);
    d->m_MultiViewer->SetShow2DCursors(show2DCursors);
    d->m_MultiViewer->SetDirectionAnnotationsVisible(showDirectionAnnotations);
    d->m_MultiViewer->SetShow3DWindowIn2x2WindowLayout(show3DWindowInMultiWindowLayout);
    d->m_MultiViewer->SetShowMagnificationSlider(showMagnificationSlider);
    d->m_MultiViewer->SetRememberSettingsPerWindowLayout(rememberSettingsPerLayout);
    d->m_MultiViewer->SetSliceIndexTracking(sliceIndexTracking);
    d->m_MultiViewer->SetTimeStepTracking(timeStepTracking);
    d->m_MultiViewer->SetMagnificationTracking(magnificationTracking);
    d->m_MultiViewer->SetDefaultWindowLayout(defaultLayout);

    this->GetSite()->GetPage()->AddPartListener(berry::IPartListener::Pointer(d->m_PartListener));

    QGridLayout *gridLayout = new QGridLayout(parent);
    gridLayout->addWidget(d->m_MultiViewer, 0, 0);
    gridLayout->setContentsMargins(0, 0, 0, 0);
    gridLayout->setSpacing(0);

    prefs->OnChanged.AddListener( berry::MessageDelegate1<niftkMultiViewerEditor, const berry::IBerryPreferences*>( this, &niftkMultiViewerEditor::OnPreferencesChanged ) );
    this->OnPreferencesChanged(prefs.GetPointer());
  }
}


//-----------------------------------------------------------------------------
niftkMultiViewerWidget* niftkMultiViewerEditor::GetMultiViewer()
{
  return d->m_MultiViewer;
}


//-----------------------------------------------------------------------------
void niftkMultiViewerEditor::SetFocus()
{
  if (d->m_MultiViewer != 0)
  {
    d->m_MultiViewer->SetFocus();
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerEditor::OnPreferencesChanged( const berry::IBerryPreferences* prefs )
{
  if (d->m_MultiViewer != NULL)
  {
    QString backgroundColourName = QString::fromStdString (prefs->GetByteArray(niftkMultiViewerEditorPreferencePage::MIDAS_BACKGROUND_COLOUR, "black"));
    QColor backgroundColour(backgroundColourName);
    d->m_MultiViewer->SetBackgroundColour(backgroundColour);
    d->m_MultiViewer->SetInterpolationType((DnDDisplayInterpolationType)(prefs->GetInt(niftkMultiViewerEditorPreferencePage::DEFAULT_INTERPOLATION_TYPE, 2)));
    d->m_MultiViewer->SetDefaultWindowLayout((WindowLayout)(prefs->GetInt(niftkMultiViewerEditorPreferencePage::MIDAS_DEFAULT_WINDOW_LAYOUT, 2))); // default coronal
    d->m_MultiViewer->SetDropType((DnDDisplayDropType)(prefs->GetInt(niftkMultiViewerEditorPreferencePage::DEFAULT_DROP_TYPE, 0)));
    d->m_MultiViewer->SetShowDropTypeControls(prefs->GetBool(niftkMultiViewerEditorPreferencePage::MIDAS_SHOW_DROP_TYPE_CONTROLS, false));
    d->m_MultiViewer->SetShowOptionsVisible(prefs->GetBool(niftkMultiViewerEditorPreferencePage::MIDAS_SHOW_SHOWING_OPTIONS, true));
    d->m_MultiViewer->SetWindowLayoutControlsVisible(prefs->GetBool(niftkMultiViewerEditorPreferencePage::MIDAS_SHOW_WINDOW_LAYOUT_CONTROLS, true));
    d->m_MultiViewer->SetViewerNumberControlsVisible(prefs->GetBool(niftkMultiViewerEditorPreferencePage::MIDAS_SHOW_VIEW_NUMBER_CONTROLS, true));
    d->m_MultiViewer->SetShowMagnificationSlider(prefs->GetBool(niftkMultiViewerEditorPreferencePage::MIDAS_SHOW_MAGNIFICATION_SLIDER, true));
    d->m_MultiViewer->SetShow2DCursors(prefs->GetBool(niftkMultiViewerEditorPreferencePage::MIDAS_SHOW_2D_CURSORS, true));
    d->m_MultiViewer->SetDirectionAnnotationsVisible(prefs->GetBool(niftkMultiViewerEditorPreferencePage::MIDAS_SHOW_DIRECTION_ANNOTATIONS, true));
    d->m_MultiViewer->SetShow3DWindowIn2x2WindowLayout(prefs->GetBool(niftkMultiViewerEditorPreferencePage::MIDAS_SHOW_3D_WINDOW_IN_MULTI_WINDOW_LAYOUT, false));
    d->m_MultiViewer->SetRememberSettingsPerWindowLayout(prefs->GetBool(niftkMultiViewerEditorPreferencePage::MIDAS_REMEMBER_VIEW_SETTINGS_PER_WINDOW_LAYOUT, true));
    d->m_MultiViewer->SetSliceIndexTracking(prefs->GetBool(niftkMultiViewerEditorPreferencePage::MIDAS_SLICE_SELECT_TRACKING, true));
    d->m_MultiViewer->SetTimeStepTracking(prefs->GetBool(niftkMultiViewerEditorPreferencePage::MIDAS_TIME_SELECT_TRACKING, true));
    d->m_MultiViewer->SetMagnificationTracking(prefs->GetBool(niftkMultiViewerEditorPreferencePage::MIDAS_MAGNIFICATION_SELECT_TRACKING, true));
  }
}

//-----------------------------------------------------------------------------
// -------------------  mitk::IRenderWindowPart  ------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
QmitkRenderWindow *niftkMultiViewerEditor::GetActiveQmitkRenderWindow() const
{
  QmitkRenderWindow* activeRenderWindow = d->m_MultiViewer->GetSelectedRenderWindow();
  if (!activeRenderWindow)
  {
    niftkSingleViewerWidget* selectedViewer = d->m_MultiViewer->GetSelectedViewer();
    activeRenderWindow = selectedViewer->GetAxialWindow();
  }
  return activeRenderWindow;
}


//-----------------------------------------------------------------------------
QHash<QString, QmitkRenderWindow *> niftkMultiViewerEditor::GetQmitkRenderWindows() const
{
  return d->m_MultiViewer->GetRenderWindows();
}


//-----------------------------------------------------------------------------
QmitkRenderWindow *niftkMultiViewerEditor::GetQmitkRenderWindow(const QString &id) const
{
  return d->m_MultiViewer->GetRenderWindow(id);
}


//-----------------------------------------------------------------------------
mitk::Point3D niftkMultiViewerEditor::GetSelectedPosition(const QString& id) const
{
  return d->m_MultiViewer->GetSelectedPosition(id);
}


//-----------------------------------------------------------------------------
void niftkMultiViewerEditor::SetSelectedPosition(const mitk::Point3D &position, const QString& id)
{
  return d->m_MultiViewer->SetSelectedPosition(position, id);
}


//-----------------------------------------------------------------------------
void niftkMultiViewerEditor::EnableDecorations(bool enable, const QStringList &decorations)
{
  // Deliberately do nothing. ToDo - maybe get niftkMultiViewerWidget to support it.
}


//-----------------------------------------------------------------------------
bool niftkMultiViewerEditor::IsDecorationEnabled(const QString &decoration) const
{
  // Deliberately deny having any decorations. ToDo - maybe get niftkMultiViewerWidget to support it.
  return false;
}


//-----------------------------------------------------------------------------
QStringList niftkMultiViewerEditor::GetDecorations() const
{
  // Deliberately return nothing. ToDo - maybe get niftkMultiViewerWidget to support it.
  QStringList decorations;
  return decorations;
}


//-----------------------------------------------------------------------------
mitk::IRenderingManager* niftkMultiViewerEditor::GetRenderingManager() const
{
  return d->m_RenderingManagerInterface;
}


//-----------------------------------------------------------------------------
mitk::SlicesRotator* niftkMultiViewerEditor::GetSlicesRotator() const
{
  // Deliberately return nothing. ToDo - maybe get niftkMultiViewerWidget to support it.
  return NULL;
}


//-----------------------------------------------------------------------------
mitk::SlicesSwiveller* niftkMultiViewerEditor::GetSlicesSwiveller() const
{
  // Deliberately return nothing. ToDo - maybe get niftkMultiViewerWidget to support it.
  return NULL;
}


//-----------------------------------------------------------------------------
void niftkMultiViewerEditor::EnableSlicingPlanes(bool enable)
{
  // Deliberately do nothing. ToDo - maybe get niftkMultiViewerWidget to support it.
  Q_UNUSED(enable);
}


//-----------------------------------------------------------------------------
bool niftkMultiViewerEditor::IsSlicingPlanesEnabled() const
{
  // Deliberately do nothing. ToDo - maybe get niftkMultiViewerWidget to support it.
  return false;
}


//-----------------------------------------------------------------------------
void niftkMultiViewerEditor::EnableLinkedNavigation(bool enable)
{
  d->m_MultiViewer->EnableLinkedNavigation(enable);
}


//-----------------------------------------------------------------------------
bool niftkMultiViewerEditor::IsLinkedNavigationEnabled() const
{
  return d->m_MultiViewer->IsLinkedNavigationEnabled();
}
