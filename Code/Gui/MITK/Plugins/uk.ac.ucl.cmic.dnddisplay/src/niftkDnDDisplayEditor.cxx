/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkDnDDisplayEditor.h"

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
#include "niftkDnDDisplayPreferencePage.h"

const std::string niftkDnDDisplayEditor::EDITOR_ID = "org.mitk.editors.dnddisplay";

class niftkDnDDisplayEditorPrivate
{
public:
  niftkDnDDisplayEditorPrivate();
  ~niftkDnDDisplayEditorPrivate();

  niftkMultiViewerWidget* m_MultiViewer;
  niftkMultiViewerVisibilityManager* m_MultiViewerVisibilityManager;
  mitk::RenderingManager::Pointer m_RenderingManager;
  berry::IPartListener::Pointer m_PartListener;
  mitk::IRenderingManager* m_RenderingManagerInterface;
};

//-----------------------------------------------------------------------------
struct niftkDnDDisplayEditorPartListener : public berry::IPartListener
{
  berryObjectMacro(niftkDnDDisplayEditorPartListener)

  //-----------------------------------------------------------------------------
  niftkDnDDisplayEditorPartListener(niftkDnDDisplayEditorPrivate* dd)
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
    if (partRef->GetId() == niftkDnDDisplayEditor::EDITOR_ID)
    {
      niftkDnDDisplayEditor::Pointer dndDisplayEditor = partRef->GetPart(false).Cast<niftkDnDDisplayEditor>();

      if (dndDisplayEditor.IsNotNull()
        && dndDisplayEditor->GetMultiViewer() == d->m_MultiViewer)
      {
        d->m_MultiViewer->Deactivated();
      }
    }
  }


  //-----------------------------------------------------------------------------
  void PartHidden (berry::IWorkbenchPartReference::Pointer partRef)
  {
    if (partRef->GetId() == niftkDnDDisplayEditor::EDITOR_ID)
    {
      niftkDnDDisplayEditor::Pointer dndDisplayEditor = partRef->GetPart(false).Cast<niftkDnDDisplayEditor>();

      if (dndDisplayEditor.IsNotNull()
        && dndDisplayEditor->GetMultiViewer() == d->m_MultiViewer)
      {
        d->m_MultiViewer->Deactivated();
      }
    }
  }


  //-----------------------------------------------------------------------------
  void PartVisible (berry::IWorkbenchPartReference::Pointer partRef)
  {
    if (partRef->GetId() == niftkDnDDisplayEditor::EDITOR_ID)
    {
      niftkDnDDisplayEditor::Pointer dndDisplayEditor = partRef->GetPart(false).Cast<niftkDnDDisplayEditor>();

      if (dndDisplayEditor.IsNotNull()
        && dndDisplayEditor->GetMultiViewer() == d->m_MultiViewer)
      {
        d->m_MultiViewer->Activated();
      }
    }
  }

private:

  niftkDnDDisplayEditorPrivate* const d;

};


//-----------------------------------------------------------------------------
niftkDnDDisplayEditorPrivate::niftkDnDDisplayEditorPrivate()
: m_MultiViewer(0)
, m_MultiViewerVisibilityManager(0)
, m_RenderingManager(0)
, m_PartListener(new niftkDnDDisplayEditorPartListener(this))
, m_RenderingManagerInterface(0)
{
  m_RenderingManager = mitk::RenderingManager::GetInstance();
  m_RenderingManager->SetConstrainedPaddingZooming(false);
  m_RenderingManagerInterface = mitk::MakeRenderingManagerInterface(m_RenderingManager);
}


//-----------------------------------------------------------------------------
niftkDnDDisplayEditorPrivate::~niftkDnDDisplayEditorPrivate()
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
niftkDnDDisplayEditor::niftkDnDDisplayEditor()
: d(new niftkDnDDisplayEditorPrivate)
{
}


//-----------------------------------------------------------------------------
niftkDnDDisplayEditor::~niftkDnDDisplayEditor()
{
  this->GetSite()->GetPage()->RemovePartListener(d->m_PartListener);
}


//-----------------------------------------------------------------------------
void niftkDnDDisplayEditor::CreateQtPartControl(QWidget* parent)
{
  if (d->m_MultiViewer == NULL)
  {
    mitk::DataStorage::Pointer dataStorage = this->GetDataStorage();
    assert(dataStorage);

    berry::IPreferencesService::Pointer prefService = berry::Platform::GetServiceRegistry().GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);
    berry::IBerryPreferences::Pointer prefs = (prefService->GetSystemPreferences()->Node(EDITOR_ID)).Cast<berry::IBerryPreferences>();
    assert( prefs );

    DnDDisplayInterpolationType defaultInterpolationType =
        (DnDDisplayInterpolationType)(prefs->GetInt(niftkDnDDisplayPreferencePage::DNDDISPLAY_DEFAULT_INTERPOLATION_TYPE, 2));
    WindowLayout defaultLayout =
        (WindowLayout)(prefs->GetInt(niftkDnDDisplayPreferencePage::DNDDISPLAY_DEFAULT_WINDOW_LAYOUT, 2)); // default = coronal
    DnDDisplayDropType defaultDropType =
        (DnDDisplayDropType)(prefs->GetInt(niftkDnDDisplayPreferencePage::DNDDISPLAY_DEFAULT_DROP_TYPE, 0));

    int defaultNumberOfRows = prefs->GetInt(niftkDnDDisplayPreferencePage::DNDDISPLAY_DEFAULT_VIEWER_ROW_NUMBER, 1);
    int defaultNumberOfColumns = prefs->GetInt(niftkDnDDisplayPreferencePage::DNDDISPLAY_DEFAULT_VIEWER_COLUMN_NUMBER, 1);
    bool showDropTypeControls = prefs->GetBool(niftkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_DROP_TYPE_CONTROLS, false);
    bool showDirectionAnnotations = prefs->GetBool(niftkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_DIRECTION_ANNOTATIONS, true);
    bool showShowingOptions = prefs->GetBool(niftkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_SHOWING_OPTIONS, true);
    bool showWindowLayoutControls = prefs->GetBool(niftkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_WINDOW_LAYOUT_CONTROLS, true);
    bool showViewerNumberControls = prefs->GetBool(niftkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_VIEWER_NUMBER_CONTROLS, true);
    bool showMagnificationSlider = prefs->GetBool(niftkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_MAGNIFICATION_SLIDER, true);
    bool show3DWindowInMultiWindowLayout = prefs->GetBool(niftkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_3D_WINDOW_IN_MULTI_WINDOW_LAYOUT, false);
    bool show2DCursors = prefs->GetBool(niftkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_2D_CURSORS, true);
    bool rememberSettingsPerLayout = prefs->GetBool(niftkDnDDisplayPreferencePage::DNDDISPLAY_REMEMBER_VIEWER_SETTINGS_PER_WINDOW_LAYOUT, true);
    bool sliceIndexTracking = prefs->GetBool(niftkDnDDisplayPreferencePage::DNDDISPLAY_SLICE_SELECT_TRACKING, true);
    bool magnificationTracking = prefs->GetBool(niftkDnDDisplayPreferencePage::DNDDISPLAY_MAGNIFICATION_SELECT_TRACKING, true);
    bool timeStepTracking = prefs->GetBool(niftkDnDDisplayPreferencePage::DNDDISPLAY_TIME_SELECT_TRACKING, true);

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

    prefs->OnChanged.AddListener( berry::MessageDelegate1<niftkDnDDisplayEditor, const berry::IBerryPreferences*>( this, &niftkDnDDisplayEditor::OnPreferencesChanged ) );
    this->OnPreferencesChanged(prefs.GetPointer());
  }
}


//-----------------------------------------------------------------------------
niftkMultiViewerWidget* niftkDnDDisplayEditor::GetMultiViewer()
{
  return d->m_MultiViewer;
}


//-----------------------------------------------------------------------------
void niftkDnDDisplayEditor::SetFocus()
{
  if (d->m_MultiViewer != 0)
  {
    d->m_MultiViewer->SetFocus();
  }
}


//-----------------------------------------------------------------------------
void niftkDnDDisplayEditor::OnPreferencesChanged( const berry::IBerryPreferences* prefs )
{
  if (d->m_MultiViewer != NULL)
  {
    QString backgroundColourName = QString::fromStdString (prefs->GetByteArray(niftkDnDDisplayPreferencePage::DNDDISPLAY_BACKGROUND_COLOUR, "black"));
    QColor backgroundColour(backgroundColourName);
    d->m_MultiViewer->SetBackgroundColour(backgroundColour);
    d->m_MultiViewer->SetInterpolationType((DnDDisplayInterpolationType)(prefs->GetInt(niftkDnDDisplayPreferencePage::DNDDISPLAY_DEFAULT_INTERPOLATION_TYPE, 2)));
    d->m_MultiViewer->SetDefaultWindowLayout((WindowLayout)(prefs->GetInt(niftkDnDDisplayPreferencePage::DNDDISPLAY_DEFAULT_WINDOW_LAYOUT, 2))); // default coronal
    d->m_MultiViewer->SetDropType((DnDDisplayDropType)(prefs->GetInt(niftkDnDDisplayPreferencePage::DNDDISPLAY_DEFAULT_DROP_TYPE, 0)));
    d->m_MultiViewer->SetShowDropTypeControls(prefs->GetBool(niftkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_DROP_TYPE_CONTROLS, false));
    d->m_MultiViewer->SetShowOptionsVisible(prefs->GetBool(niftkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_SHOWING_OPTIONS, true));
    d->m_MultiViewer->SetWindowLayoutControlsVisible(prefs->GetBool(niftkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_WINDOW_LAYOUT_CONTROLS, true));
    d->m_MultiViewer->SetViewerNumberControlsVisible(prefs->GetBool(niftkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_VIEWER_NUMBER_CONTROLS, true));
    d->m_MultiViewer->SetShowMagnificationSlider(prefs->GetBool(niftkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_MAGNIFICATION_SLIDER, true));
    d->m_MultiViewer->SetShow2DCursors(prefs->GetBool(niftkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_2D_CURSORS, true));
    d->m_MultiViewer->SetDirectionAnnotationsVisible(prefs->GetBool(niftkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_DIRECTION_ANNOTATIONS, true));
    d->m_MultiViewer->SetShow3DWindowIn2x2WindowLayout(prefs->GetBool(niftkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_3D_WINDOW_IN_MULTI_WINDOW_LAYOUT, false));
    d->m_MultiViewer->SetRememberSettingsPerWindowLayout(prefs->GetBool(niftkDnDDisplayPreferencePage::DNDDISPLAY_REMEMBER_VIEWER_SETTINGS_PER_WINDOW_LAYOUT, true));
    d->m_MultiViewer->SetSliceIndexTracking(prefs->GetBool(niftkDnDDisplayPreferencePage::DNDDISPLAY_SLICE_SELECT_TRACKING, true));
    d->m_MultiViewer->SetTimeStepTracking(prefs->GetBool(niftkDnDDisplayPreferencePage::DNDDISPLAY_TIME_SELECT_TRACKING, true));
    d->m_MultiViewer->SetMagnificationTracking(prefs->GetBool(niftkDnDDisplayPreferencePage::DNDDISPLAY_MAGNIFICATION_SELECT_TRACKING, true));
  }
}

//-----------------------------------------------------------------------------
// -------------------  mitk::IRenderWindowPart  ------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
QmitkRenderWindow *niftkDnDDisplayEditor::GetActiveQmitkRenderWindow() const
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
QHash<QString, QmitkRenderWindow *> niftkDnDDisplayEditor::GetQmitkRenderWindows() const
{
  return d->m_MultiViewer->GetRenderWindows();
}


//-----------------------------------------------------------------------------
QmitkRenderWindow *niftkDnDDisplayEditor::GetQmitkRenderWindow(const QString &id) const
{
  return d->m_MultiViewer->GetRenderWindow(id);
}


//-----------------------------------------------------------------------------
mitk::Point3D niftkDnDDisplayEditor::GetSelectedPosition(const QString& id) const
{
  return d->m_MultiViewer->GetSelectedPosition(id);
}


//-----------------------------------------------------------------------------
void niftkDnDDisplayEditor::SetSelectedPosition(const mitk::Point3D &position, const QString& id)
{
  return d->m_MultiViewer->SetSelectedPosition(position, id);
}


//-----------------------------------------------------------------------------
void niftkDnDDisplayEditor::EnableDecorations(bool enable, const QStringList &decorations)
{
  // Deliberately do nothing. ToDo - maybe get niftkMultiViewerWidget to support it.
}


//-----------------------------------------------------------------------------
bool niftkDnDDisplayEditor::IsDecorationEnabled(const QString &decoration) const
{
  // Deliberately deny having any decorations. ToDo - maybe get niftkMultiViewerWidget to support it.
  return false;
}


//-----------------------------------------------------------------------------
QStringList niftkDnDDisplayEditor::GetDecorations() const
{
  // Deliberately return nothing. ToDo - maybe get niftkMultiViewerWidget to support it.
  QStringList decorations;
  return decorations;
}


//-----------------------------------------------------------------------------
mitk::IRenderingManager* niftkDnDDisplayEditor::GetRenderingManager() const
{
  return d->m_RenderingManagerInterface;
}


//-----------------------------------------------------------------------------
mitk::SlicesRotator* niftkDnDDisplayEditor::GetSlicesRotator() const
{
  // Deliberately return nothing. ToDo - maybe get niftkMultiViewerWidget to support it.
  return NULL;
}


//-----------------------------------------------------------------------------
mitk::SlicesSwiveller* niftkDnDDisplayEditor::GetSlicesSwiveller() const
{
  // Deliberately return nothing. ToDo - maybe get niftkMultiViewerWidget to support it.
  return NULL;
}


//-----------------------------------------------------------------------------
void niftkDnDDisplayEditor::EnableSlicingPlanes(bool enable)
{
  // Deliberately do nothing. ToDo - maybe get niftkMultiViewerWidget to support it.
  Q_UNUSED(enable);
}


//-----------------------------------------------------------------------------
bool niftkDnDDisplayEditor::IsSlicingPlanesEnabled() const
{
  // Deliberately do nothing. ToDo - maybe get niftkMultiViewerWidget to support it.
  return false;
}


//-----------------------------------------------------------------------------
void niftkDnDDisplayEditor::EnableLinkedNavigation(bool enable)
{
  d->m_MultiViewer->EnableLinkedNavigation(enable);
}


//-----------------------------------------------------------------------------
bool niftkDnDDisplayEditor::IsLinkedNavigationEnabled() const
{
  return d->m_MultiViewer->IsLinkedNavigationEnabled();
}
