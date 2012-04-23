/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-12-15 15:03:56 +0000 (Thu, 15 Dec 2011) $
 Revision          : $Revision: 8030 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "QmitkMIDASMultiViewEditor.h"

#include <berryUIException.h>
#include <berryIWorkbenchPage.h>
#include <berryIPreferencesService.h>

#include <QWidget>
#include <QGridLayout>

#include "mitkGlobalInteraction.h"
#include "mitkIDataStorageService.h"
#include "mitkNodePredicateNot.h"
#include "mitkNodePredicateProperty.h"
#include "mitkMIDASDataStorageEditorInput.h"

#include "QmitkMIDASMultiViewWidget.h"
#include "QmitkMIDASMultiViewVisibilityManager.h"
#include "QmitkMIDASMultiViewEditorPreferencePage.h"

const std::string QmitkMIDASMultiViewEditor::EDITOR_ID = "org.mitk.editors.midasmultiview";

QmitkMIDASMultiViewEditor::QmitkMIDASMultiViewEditor()
 :
  m_KeyPressStateMachine(NULL)
, m_MIDASMultiViewWidget(NULL)
, m_MidasMultiViewVisibilityManager(NULL)
, m_RenderingManager(NULL)
{
  m_RenderingManager = mitk::RenderingManager::GetInstance();
  m_RenderingManager->SetConstrainedPaddingZooming(false);
}

QmitkMIDASMultiViewEditor::QmitkMIDASMultiViewEditor(const QmitkMIDASMultiViewEditor& other)
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}

QmitkMIDASMultiViewEditor::~QmitkMIDASMultiViewEditor()
{
  // we need to wrap the RemovePartListener call inside a
  // register/unregister block to prevent infinite recursion
  // due to the destruction of temporary smartpointer to this
  this->Register();
  this->GetSite()->GetPage()->RemovePartListener(berry::IPartListener::Pointer(this));
  this->UnRegister(false);

  if (m_MidasMultiViewVisibilityManager != NULL)
  {
    delete m_MidasMultiViewVisibilityManager;
  }
}

QmitkMIDASMultiViewWidget* QmitkMIDASMultiViewEditor::GetMIDASMultiViewWidget()
{
  return m_MIDASMultiViewWidget;
}

void QmitkMIDASMultiViewEditor::CreateQtPartControl(QWidget* parent)
{
  if (m_MIDASMultiViewWidget == NULL)
  {
    mitk::DataStorage::Pointer dataStorage = this->GetDataStorage();
    assert(dataStorage);

    berry::IPreferencesService::Pointer prefService = berry::Platform::GetServiceRegistry().GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);
    berry::IBerryPreferences::Pointer prefs = (prefService->GetSystemPreferences()->Node(EDITOR_ID)).Cast<berry::IBerryPreferences>();
    assert( prefs );

    MIDASDefaultInterpolationType defaultInterpolation =
        (MIDASDefaultInterpolationType)(prefs->GetInt(QmitkMIDASMultiViewEditorPreferencePage::MIDAS_DEFAULT_IMAGE_INTERPOLATION, 2));
    MIDASView defaultView =
        (MIDASView)(prefs->GetInt(QmitkMIDASMultiViewEditorPreferencePage::MIDAS_DEFAULT_VIEW, 3));
    MIDASDropType defaultDropType =
        (MIDASDropType)(prefs->GetInt(QmitkMIDASMultiViewEditorPreferencePage::MIDAS_DEFAULT_DROP_TYPE, 0));

    int defaultNumberOfRows = prefs->GetInt(QmitkMIDASMultiViewEditorPreferencePage::MIDAS_DEFAULT_NUMBER_ROWS, 1);
    int defaultNumberOfColumns = prefs->GetInt(QmitkMIDASMultiViewEditorPreferencePage::MIDAS_DEFAULT_NUMBER_COLUMNS, 1);
    bool showDropTypeWidgets = prefs->GetBool(QmitkMIDASMultiViewEditorPreferencePage::MIDAS_SHOW_DROP_TYPE_WIDGETS, false);
    bool showLayoutButtons = prefs->GetBool(QmitkMIDASMultiViewEditorPreferencePage::MIDAS_SHOW_LAYOUT_BUTTONS, false);
    bool showMagnificationSlider = prefs->GetBool(QmitkMIDASMultiViewEditorPreferencePage::MIDAS_SHOW_MAGNIFICATION_SLIDER, false);
    bool show3DViewInOrthoView = prefs->GetBool(QmitkMIDASMultiViewEditorPreferencePage::MIDAS_SHOW_3D_VIEW_IN_ORTHOVIEW, false);
    bool show2DCursors = prefs->GetBool(QmitkMIDASMultiViewEditorPreferencePage::MIDAS_SHOW_2D_CURSORS, true);
    bool rememberViewSettingsPerOrientation = prefs->GetBool(QmitkMIDASMultiViewEditorPreferencePage::MIDAS_REMEMBER_VIEW_SETTINGS_PER_ORIENTATION, false);

    m_MidasMultiViewVisibilityManager = new QmitkMIDASMultiViewVisibilityManager(dataStorage);
    m_MidasMultiViewVisibilityManager->SetDefaultInterpolationType(defaultInterpolation);
    m_MidasMultiViewVisibilityManager->SetDefaultViewType(defaultView);
    m_MidasMultiViewVisibilityManager->SetDropType(defaultDropType);

    m_RenderingManager->SetDataStorage(dataStorage);

    // Create the QmitkMIDASMultiViewWidget
    m_MIDASMultiViewWidget = new QmitkMIDASMultiViewWidget(
        m_MidasMultiViewVisibilityManager,
        m_RenderingManager,
        dataStorage,
        defaultNumberOfRows,
        defaultNumberOfColumns,
        parent);

    // Setup GUI a bit more.
    m_MIDASMultiViewWidget->SetDropTypeWidget(defaultDropType);
    m_MIDASMultiViewWidget->SetShowDropTypeWidgets(showDropTypeWidgets);
    m_MIDASMultiViewWidget->SetShowLayoutButtons(showLayoutButtons);
    m_MIDASMultiViewWidget->SetShow2DCursors(show2DCursors);
    m_MIDASMultiViewWidget->SetShow3DViewInOrthoView(show3DViewInOrthoView);
    m_MIDASMultiViewWidget->SetShowMagnificationSlider(showMagnificationSlider);
    m_MIDASMultiViewWidget->SetRememberViewSettingsPerOrientation(rememberViewSettingsPerOrientation);

    this->GetSite()->GetPage()->AddPartListener(berry::IPartListener::Pointer(this));

    QGridLayout *gridLayout = new QGridLayout(parent);
    gridLayout->addWidget(m_MIDASMultiViewWidget, 0, 0);
    gridLayout->setContentsMargins(0, 0, 0, 0);
    gridLayout->setSpacing(0);

    prefs->OnChanged.AddListener( berry::MessageDelegate1<QmitkMIDASMultiViewEditor, const berry::IBerryPreferences*>( this, &QmitkMIDASMultiViewEditor::OnPreferencesChanged ) );
    this->OnPreferencesChanged(prefs.GetPointer());

    // Create/Connect the state machine
    m_KeyPressStateMachine = mitk::MIDASKeyPressStateMachine::New("MIDASKeyPressStateMachine", m_MIDASMultiViewWidget);
    mitk::GlobalInteraction::GetInstance()->AddListener( m_KeyPressStateMachine );
  }
}

void QmitkMIDASMultiViewEditor::SetFocus()
{
  if (m_MIDASMultiViewWidget != NULL)
  {
    m_MIDASMultiViewWidget->setFocus();
  }
}

void QmitkMIDASMultiViewEditor::OnPreferencesChanged( const berry::IBerryPreferences* prefs )
{
  if (m_MIDASMultiViewWidget != NULL)
  {
    QString backgroundColourName = QString::fromStdString (prefs->GetByteArray(QmitkMIDASMultiViewEditorPreferencePage::MIDAS_BACKGROUND_COLOUR, ""));
    QColor backgroundColour(backgroundColourName);
    mitk::Color bgColour;
    if (backgroundColourName=="") // default values
    {
      bgColour[0] = 0;
      bgColour[1] = 0;
      bgColour[2] = 0;
    }
    else
    {
      bgColour[0] = backgroundColour.red() / 255.0;
      bgColour[1] = backgroundColour.green() / 255.0;
      bgColour[2] = backgroundColour.blue() / 255.0;
    }
    m_MIDASMultiViewWidget->SetBackgroundColour(bgColour);
    m_MIDASMultiViewWidget->SetDefaultInterpolationType((MIDASDefaultInterpolationType)(prefs->GetInt(QmitkMIDASMultiViewEditorPreferencePage::MIDAS_DEFAULT_IMAGE_INTERPOLATION, 2)));
    m_MIDASMultiViewWidget->SetDefaultViewType((MIDASView)(prefs->GetInt(QmitkMIDASMultiViewEditorPreferencePage::MIDAS_DEFAULT_VIEW, 3)));
    m_MIDASMultiViewWidget->SetDropTypeWidget((MIDASDropType)(prefs->GetInt(QmitkMIDASMultiViewEditorPreferencePage::MIDAS_DEFAULT_DROP_TYPE, 0)));
    m_MIDASMultiViewWidget->SetShowDropTypeWidgets(prefs->GetBool(QmitkMIDASMultiViewEditorPreferencePage::MIDAS_SHOW_DROP_TYPE_WIDGETS, false));
    m_MIDASMultiViewWidget->SetShowLayoutButtons(prefs->GetBool(QmitkMIDASMultiViewEditorPreferencePage::MIDAS_SHOW_LAYOUT_BUTTONS, false));
    m_MIDASMultiViewWidget->SetShowMagnificationSlider(prefs->GetBool(QmitkMIDASMultiViewEditorPreferencePage::MIDAS_SHOW_MAGNIFICATION_SLIDER, false));
    m_MIDASMultiViewWidget->SetShow2DCursors(prefs->GetBool(QmitkMIDASMultiViewEditorPreferencePage::MIDAS_SHOW_2D_CURSORS, true));
    m_MIDASMultiViewWidget->SetShow3DViewInOrthoView(prefs->GetBool(QmitkMIDASMultiViewEditorPreferencePage::MIDAS_SHOW_3D_VIEW_IN_ORTHOVIEW, false));
    m_MIDASMultiViewWidget->SetRememberViewSettingsPerOrientation(prefs->GetBool(QmitkMIDASMultiViewEditorPreferencePage::MIDAS_REMEMBER_VIEW_SETTINGS_PER_ORIENTATION, false));
  }
}

// -------------------  mitk::IPartListener  ----------------------

berry::IPartListener::Events::Types QmitkMIDASMultiViewEditor::GetPartEventTypes() const
{
  return Events::CLOSED | Events::HIDDEN | Events::VISIBLE;
}

void QmitkMIDASMultiViewEditor::PartClosed( berry::IWorkbenchPartReference::Pointer partRef )
{
  if (partRef->GetId() == QmitkMIDASMultiViewEditor::EDITOR_ID)
  {
    QmitkMIDASMultiViewEditor::Pointer midasMultiViewEditor = partRef->GetPart(false).Cast<QmitkMIDASMultiViewEditor>();

    if (m_MIDASMultiViewWidget == midasMultiViewEditor->GetMIDASMultiViewWidget())
    {
      m_MIDASMultiViewWidget->Deactivated();
      m_MIDASMultiViewWidget->setEnabled(false);
    }
  }
}

void QmitkMIDASMultiViewEditor::PartVisible( berry::IWorkbenchPartReference::Pointer partRef )
{
  if (partRef->GetId() == QmitkMIDASMultiViewEditor::EDITOR_ID)
  {
    QmitkMIDASMultiViewEditor::Pointer midasMultiViewEditor = partRef->GetPart(false).Cast<QmitkMIDASMultiViewEditor>();

    if (m_MIDASMultiViewWidget == midasMultiViewEditor->GetMIDASMultiViewWidget())
    {
      m_MIDASMultiViewWidget->Activated();
      m_MIDASMultiViewWidget->setEnabled(true);
    }
  }
}

void QmitkMIDASMultiViewEditor::PartHidden( berry::IWorkbenchPartReference::Pointer partRef )
{
  if (partRef->GetId() == QmitkMIDASMultiViewEditor::EDITOR_ID)
  {
    QmitkMIDASMultiViewEditor::Pointer midasMultiViewEditor = partRef->GetPart(false).Cast<QmitkMIDASMultiViewEditor>();

    if (m_MIDASMultiViewWidget == midasMultiViewEditor->GetMIDASMultiViewWidget())
    {
      m_MIDASMultiViewWidget->Deactivated();
      m_MIDASMultiViewWidget->setEnabled(false);
    }
  }
}

// -------------------  mitk::IRenderWindowPart  ----------------------

QmitkRenderWindow *QmitkMIDASMultiViewEditor::GetActiveRenderWindow() const
{
  return m_MIDASMultiViewWidget->GetActiveRenderWindow();
}

QHash<QString, QmitkRenderWindow *> QmitkMIDASMultiViewEditor::GetRenderWindows() const
{
  return m_MIDASMultiViewWidget->GetRenderWindows();
}

QmitkRenderWindow *QmitkMIDASMultiViewEditor::GetRenderWindow(const QString &id) const
{
  return m_MIDASMultiViewWidget->GetRenderWindow(id);
}

mitk::Point3D QmitkMIDASMultiViewEditor::GetSelectedPosition(const QString &id) const
{
  return m_MIDASMultiViewWidget->GetSelectedPosition(id);
}

void QmitkMIDASMultiViewEditor::SetSelectedPosition(const mitk::Point3D &pos, const QString &id)
{
  return m_MIDASMultiViewWidget->SetSelectedPosition(pos, id);
}

void QmitkMIDASMultiViewEditor::EnableDecorations(bool enable, const QStringList &decorations)
{
  // Deliberately do nothing.
}

bool QmitkMIDASMultiViewEditor::IsDecorationEnabled(const QString &decoration) const
{
  // Deliberately deny having any decorations.
  return false;
}

QStringList QmitkMIDASMultiViewEditor::GetDecorations() const
{
  // Deliberately return nothing.
  QStringList decorations;
  return decorations;
}

mitk::IRenderingManager* QmitkMIDASMultiViewEditor::GetRenderingManager() const
{
  return mitk::MakeRenderingManagerInterface(m_RenderingManager);
}
