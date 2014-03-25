/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkSingleViewerEditor.h"

#include <berryUIException.h>
#include <berryIWorkbenchPage.h>
#include <berryIPreferencesService.h>

#include <QWidget>
#include <QGridLayout>

#include <mitkGlobalInteraction.h>
#include <mitkIDataStorageService.h>
#include <mitkNodePredicateNot.h>
#include <mitkNodePredicateProperty.h>

#include <niftkSingleViewerWidget.h>
#include <niftkMultiViewerVisibilityManager.h>

#include "QmitkDnDDisplayPreferencePage.h"

const std::string QmitkSingleViewerEditor::EDITOR_ID = "org.mitk.editors.dndsingleviewer";

class QmitkSingleViewerEditorPrivate
{
public:
  QmitkSingleViewerEditorPrivate();
  ~QmitkSingleViewerEditorPrivate();

  niftkSingleViewerWidget* m_SingleViewer;
  niftkMultiViewerVisibilityManager* m_VisibilityManager;
  mitk::RenderingManager::Pointer m_RenderingManager;
  berry::IPartListener::Pointer m_PartListener;
  mitk::IRenderingManager* m_RenderingManagerInterface;
};

//-----------------------------------------------------------------------------
struct QmitkSingleViewerEditorPartListener : public berry::IPartListener
{
  berryObjectMacro(QmitkSingleViewerEditorPartListener)

  //-----------------------------------------------------------------------------
  QmitkSingleViewerEditorPartListener(QmitkSingleViewerEditorPrivate* dd)
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
    if (partRef->GetId() == QmitkSingleViewerEditor::EDITOR_ID)
    {
      QmitkSingleViewerEditor::Pointer dndDisplayEditor = partRef->GetPart(false).Cast<QmitkSingleViewerEditor>();

      if (dndDisplayEditor.IsNotNull()
        && dndDisplayEditor->GetSingleViewer() == d->m_SingleViewer)
      {
        d->m_SingleViewer->EnableLinkedNavigation(false);
      }
    }
  }


  //-----------------------------------------------------------------------------
  void PartHidden (berry::IWorkbenchPartReference::Pointer partRef)
  {
    if (partRef->GetId() == QmitkSingleViewerEditor::EDITOR_ID)
    {
      QmitkSingleViewerEditor::Pointer dndDisplayEditor = partRef->GetPart(false).Cast<QmitkSingleViewerEditor>();

      if (dndDisplayEditor.IsNotNull()
        && dndDisplayEditor->GetSingleViewer() == d->m_SingleViewer)
      {
        d->m_SingleViewer->EnableLinkedNavigation(false);
      }
    }
  }


  //-----------------------------------------------------------------------------
  void PartVisible (berry::IWorkbenchPartReference::Pointer partRef)
  {
    if (partRef->GetId() == QmitkSingleViewerEditor::EDITOR_ID)
    {
      QmitkSingleViewerEditor::Pointer dndDisplayEditor = partRef->GetPart(false).Cast<QmitkSingleViewerEditor>();

      if (dndDisplayEditor.IsNotNull()
        && dndDisplayEditor->GetSingleViewer() == d->m_SingleViewer)
      {
        d->m_SingleViewer->EnableLinkedNavigation(true);
      }
    }
  }

private:

  QmitkSingleViewerEditorPrivate* const d;

};


//-----------------------------------------------------------------------------
QmitkSingleViewerEditorPrivate::QmitkSingleViewerEditorPrivate()
: m_SingleViewer(0)
, m_VisibilityManager(0)
, m_RenderingManager(0)
, m_PartListener(new QmitkSingleViewerEditorPartListener(this))
, m_RenderingManagerInterface(0)
{
  m_RenderingManager = mitk::RenderingManager::GetInstance();
  m_RenderingManager->SetConstrainedPaddingZooming(false);
  m_RenderingManagerInterface = mitk::MakeRenderingManagerInterface(m_RenderingManager);
}


//-----------------------------------------------------------------------------
QmitkSingleViewerEditorPrivate::~QmitkSingleViewerEditorPrivate()
{
  if (m_VisibilityManager != NULL)
  {
    delete m_VisibilityManager;
  }

  if (m_RenderingManagerInterface != NULL)
  {
    delete m_RenderingManagerInterface;
  }
}


//-----------------------------------------------------------------------------
QmitkSingleViewerEditor::QmitkSingleViewerEditor()
: d(new QmitkSingleViewerEditorPrivate)
{
}


//-----------------------------------------------------------------------------
QmitkSingleViewerEditor::~QmitkSingleViewerEditor()
{
  this->GetSite()->GetPage()->RemovePartListener(d->m_PartListener);
}


//-----------------------------------------------------------------------------
void QmitkSingleViewerEditor::CreateQtPartControl(QWidget* parent)
{
  if (d->m_SingleViewer == NULL)
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

    bool showDirectionAnnotations = prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_DIRECTION_ANNOTATIONS, true);
    bool showShowingOptions = prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_SHOWING_OPTIONS, true);
    bool showWindowLayoutControls = prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_WINDOW_LAYOUT_CONTROLS, true);
    bool showMagnificationSlider = prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_MAGNIFICATION_SLIDER, true);
    bool show3DWindowInMultiWindowLayout = prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_3D_WINDOW_IN_MULTI_WINDOW_LAYOUT, false);
    bool show2DCursors = prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_2D_CURSORS, true);
    bool rememberSettingsPerLayout = prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_REMEMBER_VIEWER_SETTINGS_PER_WINDOW_LAYOUT, true);
    bool sliceIndexTracking = prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SLICE_SELECT_TRACKING, true);
    bool magnificationTracking = prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_MAGNIFICATION_SELECT_TRACKING, true);
    bool timeStepTracking = prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_TIME_SELECT_TRACKING, true);

    d->m_VisibilityManager = new niftkMultiViewerVisibilityManager(dataStorage);
    d->m_VisibilityManager->SetInterpolationType(defaultInterpolationType);
    d->m_VisibilityManager->SetDefaultWindowLayout(defaultLayout);

    d->m_RenderingManager->SetDataStorage(dataStorage);

    // Create the niftkSingleViewerWidget
    d->m_SingleViewer = new niftkSingleViewerWidget(parent, d->m_RenderingManager);
    d->m_SingleViewer->SetDataStorage(dataStorage);

    // Setup GUI a bit more.
//    d->m_SingleViewer->SetShowOptionsVisible(showShowingOptions);
//    d->m_SingleViewer->SetWindowLayoutControlsVisible(showWindowLayoutControls);
//    d->m_SingleViewer->SetCursorDefaultVisibility(show2DCursors);
    d->m_SingleViewer->SetDirectionAnnotationsVisible(showDirectionAnnotations);
    d->m_SingleViewer->SetShow3DWindowIn2x2WindowLayout(show3DWindowInMultiWindowLayout);
//    d->m_SingleViewer->SetShowMagnificationSlider(showMagnificationSlider);
    d->m_SingleViewer->SetRememberSettingsPerWindowLayout(rememberSettingsPerLayout);
//    d->m_SingleViewer->SetSliceTracking(sliceIndexTracking);
//    d->m_SingleViewer->SetTimeStepTracking(timeStepTracking);
//    d->m_SingleViewer->SetMagnificationTracking(magnificationTracking);
    d->m_VisibilityManager->SetDefaultWindowLayout(defaultLayout);

    d->m_VisibilityManager->RegisterViewer(d->m_SingleViewer);
    d->m_VisibilityManager->SetAllNodeVisibilityForViewer(0, false);
    d->m_VisibilityManager->connect(d->m_SingleViewer, SIGNAL(NodesDropped(niftkSingleViewerWidget*, QmitkRenderWindow*, std::vector<mitk::DataNode*>)), SLOT(OnNodesDropped(niftkSingleViewerWidget*, QmitkRenderWindow*, std::vector<mitk::DataNode*>)), Qt::DirectConnection);

    this->GetSite()->GetPage()->AddPartListener(berry::IPartListener::Pointer(d->m_PartListener));

    QGridLayout *gridLayout = new QGridLayout(parent);
    gridLayout->addWidget(d->m_SingleViewer, 0, 0);
    gridLayout->setContentsMargins(0, 0, 0, 0);
    gridLayout->setSpacing(0);

    prefs->OnChanged.AddListener( berry::MessageDelegate1<QmitkSingleViewerEditor, const berry::IBerryPreferences*>( this, &QmitkSingleViewerEditor::OnPreferencesChanged ) );
    this->OnPreferencesChanged(prefs.GetPointer());
  }
}


//-----------------------------------------------------------------------------
niftkSingleViewerWidget* QmitkSingleViewerEditor::GetSingleViewer()
{
  return d->m_SingleViewer;
}


//-----------------------------------------------------------------------------
void QmitkSingleViewerEditor::SetFocus()
{
  if (d->m_SingleViewer != 0)
  {
    d->m_SingleViewer->SetFocus();
  }
}


//-----------------------------------------------------------------------------
void QmitkSingleViewerEditor::OnPreferencesChanged( const berry::IBerryPreferences* prefs )
{
  if (d->m_SingleViewer != NULL)
  {
    QString backgroundColourName = QString::fromStdString (prefs->GetByteArray(QmitkDnDDisplayPreferencePage::DNDDISPLAY_BACKGROUND_COLOUR, "black"));
    QColor backgroundColour(backgroundColourName);
    d->m_SingleViewer->SetBackgroundColour(backgroundColour);
    d->m_VisibilityManager->SetInterpolationType((DnDDisplayInterpolationType)(prefs->GetInt(QmitkDnDDisplayPreferencePage::DNDDISPLAY_DEFAULT_INTERPOLATION_TYPE, 2)));
    d->m_VisibilityManager->SetDefaultWindowLayout((WindowLayout)(prefs->GetInt(QmitkDnDDisplayPreferencePage::DNDDISPLAY_DEFAULT_WINDOW_LAYOUT, 2))); // default coronal
//    d->m_SingleViewer->SetShowOptionsVisible(prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_SHOWING_OPTIONS, true));
//    d->m_SingleViewer->SetWindowLayoutControlsVisible(prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_WINDOW_LAYOUT_CONTROLS, true));
//    d->m_SingleViewer->SetShowMagnificationSlider(prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_MAGNIFICATION_SLIDER, true));
//    d->m_SingleViewer->SetCursorDefaultVisibility(prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_2D_CURSORS, true));
    d->m_SingleViewer->SetDirectionAnnotationsVisible(prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_DIRECTION_ANNOTATIONS, true));
    d->m_SingleViewer->SetShow3DWindowIn2x2WindowLayout(prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_3D_WINDOW_IN_MULTI_WINDOW_LAYOUT, false));
    d->m_SingleViewer->SetRememberSettingsPerWindowLayout(prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_REMEMBER_VIEWER_SETTINGS_PER_WINDOW_LAYOUT, true));
//    d->m_SingleViewer->SetSliceTracking(prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SLICE_SELECT_TRACKING, true));
//    d->m_SingleViewer->SetTimeStepTracking(prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_TIME_SELECT_TRACKING, true));
//    d->m_SingleViewer->SetMagnificationTracking(prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_MAGNIFICATION_SELECT_TRACKING, true));
  }
}

//-----------------------------------------------------------------------------
// -------------------  mitk::IRenderWindowPart  ------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
QmitkRenderWindow *QmitkSingleViewerEditor::GetActiveQmitkRenderWindow() const
{
  QmitkRenderWindow* activeRenderWindow = d->m_SingleViewer->GetSelectedRenderWindow();
  if (!activeRenderWindow)
  {
    activeRenderWindow = d->m_SingleViewer->GetVisibleRenderWindows()[0];
  }
  return activeRenderWindow;
}


//-----------------------------------------------------------------------------
QHash<QString, QmitkRenderWindow *> QmitkSingleViewerEditor::GetQmitkRenderWindows() const
{
  // NOTE: This MUST always return a non-empty map.

  QHash<QString, QmitkRenderWindow*> renderWindows;

  // See org.mitk.gui.qt.imagenavigator plugin.
  //
  // The assumption is that a QmitkStdMultiWidget has windows called
  // axial, sagittal, coronal, 3d.
  //
  // So, if we take the currently selected widget, and name these render windows
  // accordingly, then the MITK imagenavigator can be used to update it.

  renderWindows.insert("axial", d->m_SingleViewer->GetAxialWindow());
  renderWindows.insert("sagittal", d->m_SingleViewer->GetSagittalWindow());
  renderWindows.insert("coronal", d->m_SingleViewer->GetCoronalWindow());
  renderWindows.insert("3d", d->m_SingleViewer->Get3DWindow());

  return renderWindows;
}


//-----------------------------------------------------------------------------
QmitkRenderWindow *QmitkSingleViewerEditor::GetQmitkRenderWindow(const QString &id) const
{
  return this->GetQmitkRenderWindows()[id];
}


//-----------------------------------------------------------------------------
mitk::Point3D QmitkSingleViewerEditor::GetSelectedPosition(const QString& id) const
{
  if (id.isNull() || this->GetQmitkRenderWindow(id))
  {
    return d->m_SingleViewer->GetSelectedPosition();
  }

  mitk::Point3D fallBackValue;
  fallBackValue.Fill(0.0);
  return fallBackValue;
}


//-----------------------------------------------------------------------------
void QmitkSingleViewerEditor::SetSelectedPosition(const mitk::Point3D& position, const QString& id)
{
  if (id.isNull() || this->GetQmitkRenderWindow(id))
  {
    d->m_SingleViewer->SetSelectedPosition(position);
  }
}


//-----------------------------------------------------------------------------
void QmitkSingleViewerEditor::EnableDecorations(bool enable, const QStringList &decorations)
{
  // Deliberately do nothing. ToDo - maybe get niftkSingleViewerWidget to support it.
}


//-----------------------------------------------------------------------------
bool QmitkSingleViewerEditor::IsDecorationEnabled(const QString &decoration) const
{
  // Deliberately deny having any decorations. ToDo - maybe get niftkSingleViewerWidget to support it.
  return false;
}


//-----------------------------------------------------------------------------
QStringList QmitkSingleViewerEditor::GetDecorations() const
{
  // Deliberately return nothing. ToDo - maybe get niftkSingleViewerWidget to support it.
  QStringList decorations;
  return decorations;
}


//-----------------------------------------------------------------------------
mitk::IRenderingManager* QmitkSingleViewerEditor::GetRenderingManager() const
{
  return d->m_RenderingManagerInterface;
}


//-----------------------------------------------------------------------------
mitk::SlicesRotator* QmitkSingleViewerEditor::GetSlicesRotator() const
{
  // Deliberately return nothing. ToDo - maybe get niftkSingleViewerWidget to support it.
  return NULL;
}


//-----------------------------------------------------------------------------
mitk::SlicesSwiveller* QmitkSingleViewerEditor::GetSlicesSwiveller() const
{
  // Deliberately return nothing. ToDo - maybe get niftkSingleViewerWidget to support it.
  return NULL;
}


//-----------------------------------------------------------------------------
void QmitkSingleViewerEditor::EnableSlicingPlanes(bool enable)
{
  // Deliberately do nothing. ToDo - maybe get niftkSingleViewerWidget to support it.
  Q_UNUSED(enable);
}


//-----------------------------------------------------------------------------
bool QmitkSingleViewerEditor::IsSlicingPlanesEnabled() const
{
  // Deliberately do nothing. ToDo - maybe get niftkSingleViewerWidget to support it.
  return false;
}


//-----------------------------------------------------------------------------
void QmitkSingleViewerEditor::EnableLinkedNavigation(bool enable)
{
  d->m_SingleViewer->EnableLinkedNavigation(enable);
}


//-----------------------------------------------------------------------------
bool QmitkSingleViewerEditor::IsLinkedNavigationEnabled() const
{
  return d->m_SingleViewer->IsLinkedNavigationEnabled();
}
