/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkMultiViewerEditor.h"

#include <berryUIException.h>
#include <berryIWorkbenchPage.h>
#include <berryIPreferencesService.h>

#include <QWidget>
#include <QGridLayout>

#include <mitkIDataStorageService.h>
#include <mitkNodePredicateNot.h>
#include <mitkNodePredicateProperty.h>

#include <niftkMultiViewerWidget.h>
#include <niftkMultiViewerVisibilityManager.h>
#include "QmitkDnDDisplayPreferencePage.h"

const std::string QmitkMultiViewerEditor::EDITOR_ID = "org.mitk.editors.dndmultidisplay";

class QmitkMultiViewerEditorPrivate
{
public:
  QmitkMultiViewerEditorPrivate();
  ~QmitkMultiViewerEditorPrivate();

  niftkMultiViewerWidget* m_MultiViewer;
  niftkMultiViewerVisibilityManager::Pointer m_MultiViewerVisibilityManager;
  mitk::RenderingManager::Pointer m_RenderingManager;
  berry::IPartListener::Pointer m_PartListener;
  mitk::IRenderingManager* m_RenderingManagerInterface;
};

//-----------------------------------------------------------------------------
struct QmitkMultiViewerEditorPartListener : public berry::IPartListener
{
  berryObjectMacro(QmitkMultiViewerEditorPartListener)

  //-----------------------------------------------------------------------------
  QmitkMultiViewerEditorPartListener(QmitkMultiViewerEditorPrivate* dd)
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
    if (partRef->GetId() == QmitkMultiViewerEditor::EDITOR_ID)
    {
      QmitkMultiViewerEditor::Pointer dndDisplayEditor = partRef->GetPart(false).Cast<QmitkMultiViewerEditor>();

      if (dndDisplayEditor.IsNotNull()
        && dndDisplayEditor->GetMultiViewer() == d->m_MultiViewer)
      {
        d->m_MultiViewer->EnableLinkedNavigation(false);
      }
    }
  }


  //-----------------------------------------------------------------------------
  void PartHidden (berry::IWorkbenchPartReference::Pointer partRef)
  {
    if (partRef->GetId() == QmitkMultiViewerEditor::EDITOR_ID)
    {
      QmitkMultiViewerEditor::Pointer dndDisplayEditor = partRef->GetPart(false).Cast<QmitkMultiViewerEditor>();

      if (dndDisplayEditor.IsNotNull()
        && dndDisplayEditor->GetMultiViewer() == d->m_MultiViewer)
      {
        d->m_MultiViewer->EnableLinkedNavigation(false);
      }
    }
  }


  //-----------------------------------------------------------------------------
  void PartVisible (berry::IWorkbenchPartReference::Pointer partRef)
  {
    if (partRef->GetId() == QmitkMultiViewerEditor::EDITOR_ID)
    {
      QmitkMultiViewerEditor::Pointer dndDisplayEditor = partRef->GetPart(false).Cast<QmitkMultiViewerEditor>();

      if (dndDisplayEditor.IsNotNull()
        && dndDisplayEditor->GetMultiViewer() == d->m_MultiViewer)
      {
        d->m_MultiViewer->EnableLinkedNavigation(true);
      }
    }
  }

private:

  QmitkMultiViewerEditorPrivate* const d;

};


//-----------------------------------------------------------------------------
QmitkMultiViewerEditorPrivate::QmitkMultiViewerEditorPrivate()
: m_MultiViewer(0)
, m_MultiViewerVisibilityManager(0)
, m_RenderingManager(0)
, m_PartListener(new QmitkMultiViewerEditorPartListener(this))
, m_RenderingManagerInterface(0)
{
  /// Note:
  /// The DnD Display must use its own rendering manager, not the global one
  /// returned by mitk::RenderingManager::GetInstance(). The reason is that
  /// the global rendering manager is reinitialised by its InitializeViewsByBoundingObjects
  /// function whenever a new file is opened, recalculating a new bounding box
  /// based on all the globally visible nodes in the data storage.
  m_RenderingManager = mitk::RenderingManager::New();
  m_RenderingManager->SetConstrainedPaddingZooming(false);
  m_RenderingManagerInterface = mitk::MakeRenderingManagerInterface(m_RenderingManager);
}


//-----------------------------------------------------------------------------
QmitkMultiViewerEditorPrivate::~QmitkMultiViewerEditorPrivate()
{
  if (m_RenderingManagerInterface != NULL)
  {
    delete m_RenderingManagerInterface;
  }
}


//-----------------------------------------------------------------------------
QmitkMultiViewerEditor::QmitkMultiViewerEditor()
: d(new QmitkMultiViewerEditorPrivate)
{
}


//-----------------------------------------------------------------------------
QmitkMultiViewerEditor::~QmitkMultiViewerEditor()
{
  this->GetSite()->GetPage()->RemovePartListener(d->m_PartListener);
}


//-----------------------------------------------------------------------------
void QmitkMultiViewerEditor::CreateQtPartControl(QWidget* parent)
{
  if (d->m_MultiViewer == NULL)
  {
    mitk::DataStorage::Pointer dataStorage = this->GetDataStorage();
    assert(dataStorage);

    berry::IPreferencesService::Pointer prefService = berry::Platform::GetServiceRegistry().GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);
    berry::IBerryPreferences::Pointer prefs = (prefService->GetSystemPreferences()->Node(EDITOR_ID)).Cast<berry::IBerryPreferences>();
    assert( prefs );

    DnDDisplayInterpolationType defaultInterpolationType =
        (DnDDisplayInterpolationType)(prefs->GetInt(QmitkDnDDisplayPreferencePage::DNDDISPLAY_DEFAULT_INTERPOLATION_TYPE, 2));
    WindowLayout defaultLayout =
        (WindowLayout)(prefs->GetInt(QmitkDnDDisplayPreferencePage::DNDDISPLAY_DEFAULT_WINDOW_LAYOUT, 2)); // default = coronal
    DnDDisplayDropType defaultDropType =
        (DnDDisplayDropType)(prefs->GetInt(QmitkDnDDisplayPreferencePage::DNDDISPLAY_DEFAULT_DROP_TYPE, 0));

    int defaultNumberOfRows = prefs->GetInt(QmitkDnDDisplayPreferencePage::DNDDISPLAY_DEFAULT_VIEWER_ROW_NUMBER, 1);
    int defaultNumberOfColumns = prefs->GetInt(QmitkDnDDisplayPreferencePage::DNDDISPLAY_DEFAULT_VIEWER_COLUMN_NUMBER, 1);
    bool showDropTypeControls = prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_DROP_TYPE_CONTROLS, false);
    bool showDirectionAnnotations = prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_DIRECTION_ANNOTATIONS, true);
    bool showShowingOptions = prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_SHOWING_OPTIONS, true);
    bool showWindowLayoutControls = prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_WINDOW_LAYOUT_CONTROLS, true);
    bool showViewerNumberControls = prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_VIEWER_NUMBER_CONTROLS, true);
    bool showMagnificationSlider = prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_MAGNIFICATION_SLIDER, true);
    bool show3DWindowInMultiWindowLayout = prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_3D_WINDOW_IN_MULTI_WINDOW_LAYOUT, false);
    bool show2DCursors = prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_2D_CURSORS, true);
    bool rememberSettingsPerLayout = prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_REMEMBER_VIEWER_SETTINGS_PER_WINDOW_LAYOUT, true);
    bool sliceIndexTracking = prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SLICE_SELECT_TRACKING, true);
    bool magnificationTracking = prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_MAGNIFICATION_SELECT_TRACKING, true);
    bool timeStepTracking = prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_TIME_SELECT_TRACKING, true);

    d->m_MultiViewerVisibilityManager = niftkMultiViewerVisibilityManager::New(dataStorage);
    d->m_MultiViewerVisibilityManager->SetInterpolationType(defaultInterpolationType);
    d->m_MultiViewerVisibilityManager->SetDefaultWindowLayout(defaultLayout);
    d->m_MultiViewerVisibilityManager->SetDropType(defaultDropType);

    d->m_RenderingManager->SetDataStorage(dataStorage);

    // Create the niftkMultiViewerWidget
    d->m_MultiViewer = new niftkMultiViewerWidget(
        d->m_MultiViewerVisibilityManager,
        d->m_RenderingManager,
        defaultNumberOfRows,
        defaultNumberOfColumns,
        parent);

    // Setup GUI a bit more.
    d->m_MultiViewer->SetDropType(defaultDropType);
    d->m_MultiViewer->SetShowOptionsVisible(showShowingOptions);
    d->m_MultiViewer->SetWindowLayoutControlsVisible(showWindowLayoutControls);
    d->m_MultiViewer->SetViewerNumberControlsVisible(showViewerNumberControls);
    d->m_MultiViewer->SetShowDropTypeControls(showDropTypeControls);
    d->m_MultiViewer->SetCursorDefaultVisibility(show2DCursors);
    d->m_MultiViewer->SetDirectionAnnotationsVisible(showDirectionAnnotations);
    d->m_MultiViewer->SetShow3DWindowIn2x2WindowLayout(show3DWindowInMultiWindowLayout);
    d->m_MultiViewer->SetShowMagnificationSlider(showMagnificationSlider);
    d->m_MultiViewer->SetRememberSettingsPerWindowLayout(rememberSettingsPerLayout);
    d->m_MultiViewer->SetSliceTracking(sliceIndexTracking);
    d->m_MultiViewer->SetTimeStepTracking(timeStepTracking);
    d->m_MultiViewer->SetMagnificationTracking(magnificationTracking);
    d->m_MultiViewer->SetDefaultWindowLayout(defaultLayout);

    this->GetSite()->GetPage()->AddPartListener(berry::IPartListener::Pointer(d->m_PartListener));

    QGridLayout *gridLayout = new QGridLayout(parent);
    gridLayout->addWidget(d->m_MultiViewer, 0, 0);
    gridLayout->setContentsMargins(0, 0, 0, 0);
    gridLayout->setSpacing(0);

    prefs->OnChanged.AddListener( berry::MessageDelegate1<QmitkMultiViewerEditor, const berry::IBerryPreferences*>( this, &QmitkMultiViewerEditor::OnPreferencesChanged ) );
    this->OnPreferencesChanged(prefs.GetPointer());
  }
}


//-----------------------------------------------------------------------------
niftkMultiViewerWidget* QmitkMultiViewerEditor::GetMultiViewer()
{
  return d->m_MultiViewer;
}


//-----------------------------------------------------------------------------
void QmitkMultiViewerEditor::SetFocus()
{
  if (d->m_MultiViewer != 0)
  {
    d->m_MultiViewer->SetFocused();
  }
}


//-----------------------------------------------------------------------------
void QmitkMultiViewerEditor::OnPreferencesChanged( const berry::IBerryPreferences* prefs )
{
  if (d->m_MultiViewer != NULL)
  {
    QString backgroundColourName = QString::fromStdString (prefs->GetByteArray(QmitkDnDDisplayPreferencePage::DNDDISPLAY_BACKGROUND_COLOUR, "black"));
    QColor backgroundColour(backgroundColourName);
    d->m_MultiViewer->SetBackgroundColour(backgroundColour);
    d->m_MultiViewer->SetInterpolationType((DnDDisplayInterpolationType)(prefs->GetInt(QmitkDnDDisplayPreferencePage::DNDDISPLAY_DEFAULT_INTERPOLATION_TYPE, 2)));
    d->m_MultiViewer->SetDefaultWindowLayout((WindowLayout)(prefs->GetInt(QmitkDnDDisplayPreferencePage::DNDDISPLAY_DEFAULT_WINDOW_LAYOUT, 2))); // default coronal
    d->m_MultiViewer->SetDropType((DnDDisplayDropType)(prefs->GetInt(QmitkDnDDisplayPreferencePage::DNDDISPLAY_DEFAULT_DROP_TYPE, 0)));
    d->m_MultiViewer->SetShowDropTypeControls(prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_DROP_TYPE_CONTROLS, false));
    d->m_MultiViewer->SetShowOptionsVisible(prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_SHOWING_OPTIONS, true));
    d->m_MultiViewer->SetWindowLayoutControlsVisible(prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_WINDOW_LAYOUT_CONTROLS, true));
    d->m_MultiViewer->SetViewerNumberControlsVisible(prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_VIEWER_NUMBER_CONTROLS, true));
    d->m_MultiViewer->SetShowMagnificationSlider(prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_MAGNIFICATION_SLIDER, true));
    d->m_MultiViewer->SetCursorDefaultVisibility(prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_2D_CURSORS, true));
    d->m_MultiViewer->SetDirectionAnnotationsVisible(prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_DIRECTION_ANNOTATIONS, true));
    d->m_MultiViewer->SetShow3DWindowIn2x2WindowLayout(prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_3D_WINDOW_IN_MULTI_WINDOW_LAYOUT, false));
    d->m_MultiViewer->SetRememberSettingsPerWindowLayout(prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_REMEMBER_VIEWER_SETTINGS_PER_WINDOW_LAYOUT, true));
    d->m_MultiViewer->SetSliceTracking(prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SLICE_SELECT_TRACKING, true));
    d->m_MultiViewer->SetTimeStepTracking(prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_TIME_SELECT_TRACKING, true));
    d->m_MultiViewer->SetMagnificationTracking(prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_MAGNIFICATION_SELECT_TRACKING, true));
  }
}

//-----------------------------------------------------------------------------
// -------------------  mitk::IRenderWindowPart  ------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
QmitkRenderWindow *QmitkMultiViewerEditor::GetActiveQmitkRenderWindow() const
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
QHash<QString, QmitkRenderWindow *> QmitkMultiViewerEditor::GetQmitkRenderWindows() const
{
  return d->m_MultiViewer->GetRenderWindows();
}


//-----------------------------------------------------------------------------
QmitkRenderWindow *QmitkMultiViewerEditor::GetQmitkRenderWindow(const QString &id) const
{
  return d->m_MultiViewer->GetRenderWindow(id);
}


//-----------------------------------------------------------------------------
mitk::Point3D QmitkMultiViewerEditor::GetSelectedPosition(const QString& id) const
{
  return d->m_MultiViewer->GetSelectedPosition(id);
}


//-----------------------------------------------------------------------------
void QmitkMultiViewerEditor::SetSelectedPosition(const mitk::Point3D &position, const QString& id)
{
  return d->m_MultiViewer->SetSelectedPosition(position, id);
}


//-----------------------------------------------------------------------------
void QmitkMultiViewerEditor::EnableDecorations(bool enable, const QStringList &decorations)
{
  // Deliberately do nothing. ToDo - maybe get niftkMultiViewerWidget to support it.
}


//-----------------------------------------------------------------------------
bool QmitkMultiViewerEditor::IsDecorationEnabled(const QString &decoration) const
{
  // Deliberately deny having any decorations. ToDo - maybe get niftkMultiViewerWidget to support it.
  return false;
}


//-----------------------------------------------------------------------------
QStringList QmitkMultiViewerEditor::GetDecorations() const
{
  // Deliberately return nothing. ToDo - maybe get niftkMultiViewerWidget to support it.
  QStringList decorations;
  return decorations;
}


//-----------------------------------------------------------------------------
mitk::IRenderingManager* QmitkMultiViewerEditor::GetRenderingManager() const
{
  return d->m_RenderingManagerInterface;
}


//-----------------------------------------------------------------------------
mitk::SlicesRotator* QmitkMultiViewerEditor::GetSlicesRotator() const
{
  // Deliberately return nothing. ToDo - maybe get niftkMultiViewerWidget to support it.
  return NULL;
}


//-----------------------------------------------------------------------------
mitk::SlicesSwiveller* QmitkMultiViewerEditor::GetSlicesSwiveller() const
{
  // Deliberately return nothing. ToDo - maybe get niftkMultiViewerWidget to support it.
  return NULL;
}


//-----------------------------------------------------------------------------
void QmitkMultiViewerEditor::EnableSlicingPlanes(bool enable)
{
  // Deliberately do nothing. ToDo - maybe get niftkMultiViewerWidget to support it.
  Q_UNUSED(enable);
}


//-----------------------------------------------------------------------------
bool QmitkMultiViewerEditor::IsSlicingPlanesEnabled() const
{
  // Deliberately do nothing. ToDo - maybe get niftkMultiViewerWidget to support it.
  return false;
}


//-----------------------------------------------------------------------------
void QmitkMultiViewerEditor::EnableLinkedNavigation(bool enable)
{
  d->m_MultiViewer->EnableLinkedNavigation(enable);
}


//-----------------------------------------------------------------------------
bool QmitkMultiViewerEditor::IsLinkedNavigationEnabled() const
{
  return d->m_MultiViewer->IsLinkedNavigationEnabled();
}
