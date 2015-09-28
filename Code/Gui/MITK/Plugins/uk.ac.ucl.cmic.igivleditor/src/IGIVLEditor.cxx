/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "IGIVLEditor.h"

#include <berryUIException.h>
#include <berryIWorkbenchPage.h>
#include <berryIPreferencesService.h>
#include <berryIPartListener.h>
#include <ctkPluginContext.h>
#include <ctkServiceReference.h>
#include <service/event/ctkEventConstants.h>
#include <service/event/ctkEventAdmin.h>
#include <service/event/ctkEvent.h>

#include <QWidget>
#include <QDateTime>
#include <QFile>
#include <QTextStream>
#include <QDir>

// Note:
// This header must be included before mitkOclResourceService.h to avoid name clash between Xlib.h and Qt.
// Both headers define a 'None' constant. The header below undefines it to avoid compile error with gcc.
#include <VLEditor/QmitkIGIVLEditor.h>

#include <mitkColorProperty.h>
#include <mitkGlobalInteraction.h>
#include <mitkNodePredicateNot.h>
#include <mitkNodePredicateProperty.h>
#include <mitkOclResourceService.h>

#include <mitkDataStorageEditorInput.h>
#include <mitkIDataStorageService.h>

#include "internal/IGIVLEditorPreferencePage.h"
#include "internal/IGIVLEditorActivator.h"


//-----------------------------------------------------------------------------
const char* IGIVLEditor::EDITOR_ID = "org.mitk.editors.igivleditor";


/**
 * \class IGIVLEditorPrivate
 * \brief PIMPL pattern implementation of IGIVLEditor.
 */
class IGIVLEditorPrivate
{
public:

  IGIVLEditorPrivate();
  ~IGIVLEditorPrivate();

  QmitkIGIVLEditor* m_IGIVLEditor;
  berry::IPartListener::Pointer m_PartListener;
};


/**
 * \class IGIVLEditorWidgetPartListener
 * \brief Used to handle interaction with the contained overlay
 * editor widget when this IGIOverlayEditor is opened/closed etc.
 */
struct IGIVLEditorWidgetPartListener : public berry::IPartListener
{
  berryObjectMacro(IGIVLEditorWidgetPartListener)

  //---------------------------------------------------------------------------
  IGIVLEditorWidgetPartListener(IGIVLEditorPrivate* dd)
    : d(dd)
  {}

  //---------------------------------------------------------------------------
  Events::Types GetPartEventTypes() const
  {
    return Events::CLOSED | Events::HIDDEN | Events::VISIBLE;
  }

  //---------------------------------------------------------------------------
  void PartClosed(berry::IWorkbenchPartReference::Pointer partRef)
  {
    if (partRef->GetId() == IGIVLEditor::EDITOR_ID)
    {
      IGIVLEditor::Pointer editor = partRef->GetPart(false).Cast<IGIVLEditor>();
      if (d->m_IGIVLEditor == editor->GetIGIVLEditor())
      {
        // Call editor to turn things off as the widget is being closed.
      }
    }
  }

  //---------------------------------------------------------------------------
  void PartHidden(berry::IWorkbenchPartReference::Pointer partRef)
  {
    if (partRef->GetId() == IGIVLEditor::EDITOR_ID)
    {
      IGIVLEditor::Pointer editor = partRef->GetPart(false).Cast<IGIVLEditor>();
      if (d->m_IGIVLEditor == editor->GetIGIVLEditor())
      {
        // Call editor to turn things off as the widget is being hidden.
      }
    }
  }

  //---------------------------------------------------------------------------
  void PartVisible(berry::IWorkbenchPartReference::Pointer partRef)
  {
    if (partRef->GetId() == IGIVLEditor::EDITOR_ID)
    {
      IGIVLEditor::Pointer editor = partRef->GetPart(false).Cast<IGIVLEditor>();
      if (d->m_IGIVLEditor == editor->GetIGIVLEditor())
      {
        // Call editor to turn things on as the widget is being made visible.
      }
    }
  }

private:

  IGIVLEditorPrivate* const d;
};


//-----------------------------------------------------------------------------
IGIVLEditorPrivate::IGIVLEditorPrivate()
  : m_IGIVLEditor(0)
  , m_PartListener(new IGIVLEditorWidgetPartListener(this))
{
}


//-----------------------------------------------------------------------------
IGIVLEditorPrivate::~IGIVLEditorPrivate()
{
}

//-----------------------------------------------------------------------------
IGIVLEditor::IGIVLEditor()
  : d(new IGIVLEditorPrivate)
{
}


//-----------------------------------------------------------------------------
IGIVLEditor::~IGIVLEditor()
{
  this->GetSite()->GetPage()->RemovePartListener(d->m_PartListener);
}


//-----------------------------------------------------------------------------
QmitkIGIVLEditor* IGIVLEditor::GetIGIVLEditor()
{
  return d->m_IGIVLEditor;
}


//-----------------------------------------------------------------------------
QmitkRenderWindow *IGIVLEditor::GetActiveQmitkRenderWindow() const
{
  return 0;
}


//-----------------------------------------------------------------------------
QHash<QString, QmitkRenderWindow *> IGIVLEditor::GetQmitkRenderWindows() const
{
  return QHash<QString, QmitkRenderWindow *>();
}


//-----------------------------------------------------------------------------
QmitkRenderWindow *IGIVLEditor::GetQmitkRenderWindow(const QString &id) const
{
  return 0;
}


//-----------------------------------------------------------------------------
mitk::Point3D IGIVLEditor::GetSelectedPosition(const QString & id) const
{
  // Not implemented.
  mitk::Point3D point;
  point[0] = 0;
  point[1] = 0;
  point[2] = 0;
  return point;
}


//-----------------------------------------------------------------------------
void IGIVLEditor::SetSelectedPosition(const mitk::Point3D &pos, const QString &id)
{
  // Not implemented.
}


//-----------------------------------------------------------------------------
void IGIVLEditor::EnableDecorations(bool /*enable*/, const QStringList & /*decorations*/)
{
}


//-----------------------------------------------------------------------------
bool IGIVLEditor::IsDecorationEnabled(const QString & /*decoration*/) const
{
  return false;
}


//-----------------------------------------------------------------------------
QStringList IGIVLEditor::GetDecorations() const
{
  QStringList decorations;
  return decorations;
}


//-----------------------------------------------------------------------------
mitk::SlicesRotator* IGIVLEditor::GetSlicesRotator() const
{
  return NULL;
}


//-----------------------------------------------------------------------------
mitk::SlicesSwiveller* IGIVLEditor::GetSlicesSwiveller() const
{
  return NULL;
}


//-----------------------------------------------------------------------------
void IGIVLEditor::EnableSlicingPlanes(bool /*enable*/)
{
}


//-----------------------------------------------------------------------------
bool IGIVLEditor::IsSlicingPlanesEnabled() const
{
  return false;
}


//-----------------------------------------------------------------------------
void IGIVLEditor::EnableLinkedNavigation(bool /*enable*/)
{
}


//-----------------------------------------------------------------------------
bool IGIVLEditor::IsLinkedNavigationEnabled() const
{
  return false;
}


//-----------------------------------------------------------------------------
void IGIVLEditor::CreateQtPartControl(QWidget* parent)
{
  if (d->m_IGIVLEditor == 0)
  {
    QHBoxLayout* layout = new QHBoxLayout(parent);
    layout->setContentsMargins(0,0,0,0);

    d->m_IGIVLEditor = new QmitkIGIVLEditor(parent);
    layout->addWidget(d->m_IGIVLEditor);


    ctkPluginContext*     context     = mitk::IGIVLEditorActivator::/*GetDefault()->*/getContext();
    ctkServiceReference   serviceRef  = context->getServiceReference<OclResourceService>();
    OclResourceService*   oclService  = context->getService<OclResourceService>(serviceRef);
    if (oclService == NULL)
    {
      mitkThrow() << "Failed to find OpenCL resource service." << std::endl;
    }
    d->m_IGIVLEditor->SetOclResourceService(oclService);


    mitk::DataStorage::Pointer ds = this->GetDataStorage();
    d->m_IGIVLEditor->SetDataStorage(ds);

    this->GetSite()->GetPage()->AddPartListener(d->m_PartListener);

    QMetaObject::invokeMethod(this, "OnPreferencesChanged", Qt::QueuedConnection);

    this->RequestUpdate();

    // Finally: Listen to update pulse coming off of event bus. This pulse comes from the data manager updating.
    ctkServiceReference ref = mitk::IGIVLEditorActivator::getContext()->getServiceReference<ctkEventAdmin>();
    if (ref)
    {
      ctkEventAdmin* eventAdmin = mitk::IGIVLEditorActivator::getContext()->getService<ctkEventAdmin>(ref);
      
      ctkDictionary propertiesIGI;
      propertiesIGI[ctkEventConstants::EVENT_TOPIC] = "uk/ac/ucl/cmic/IGIUPDATE";
      eventAdmin->subscribeSlot(this, SLOT(OnIGIUpdate(ctkEvent)), propertiesIGI);

      ctkDictionary propertiesRecordingStarted;
      propertiesRecordingStarted[ctkEventConstants::EVENT_TOPIC] = "uk/ac/ucl/cmic/IGIRECORDINGSTARTED";
      eventAdmin->subscribeSlot(this, SLOT(OnRecordingStarted(ctkEvent)), propertiesRecordingStarted);
    }
  }
}


//-----------------------------------------------------------------------------
void IGIVLEditor::OnPreferencesChanged()
{
  berry::IPreferencesService::Pointer prefService = berry::Platform::GetServiceRegistry().GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);
  berry::IBerryPreferences::Pointer   prefsNode   = prefService->GetSystemPreferences()->Node(EDITOR_ID).Cast<berry::IBerryPreferences>();

  this->OnPreferencesChanged(prefsNode.GetPointer());
}


//-----------------------------------------------------------------------------
void IGIVLEditor::OnPreferencesChanged(const berry::IBerryPreferences* prefs)
{
  // 0xAABBGGRR
  unsigned int   backgroundColour = prefs->GetInt(IGIVLEditorPreferencePage::BACKGROUND_COLOR_PREFSKEY, IGIVLEditorPreferencePage::DEFAULT_BACKGROUND_COLOR);

  if (d->m_IGIVLEditor != 0)
  {
    d->m_IGIVLEditor->SetBackgroundColour(backgroundColour);
  }
}


//-----------------------------------------------------------------------------
void IGIVLEditor::SetFocus()
{
  if (d->m_IGIVLEditor != 0)
  {
    d->m_IGIVLEditor->setFocus();
  }
}


//-----------------------------------------------------------------------------
void IGIVLEditor::OnIGIUpdate(const ctkEvent& event)
{
  d->m_IGIVLEditor->Update();
}


//-----------------------------------------------------------------------------
void IGIVLEditor::WriteCurrentConfig(const QString& directory) const
{
  QFile   infoFile(directory + QDir::separator() + EDITOR_ID + ".txt");
  bool opened = infoFile.open(QIODevice::ReadWrite | QIODevice::Text | QIODevice::Append);
  if (opened)
  {
    QTextStream   info(&infoFile);
    info.setCodec("UTF-8");
    info << "START: " << QDateTime::currentDateTime().toString() << "\n";
  }
}


//-----------------------------------------------------------------------------
void IGIVLEditor::OnRecordingStarted(const ctkEvent& event)
{
  QString   directory = event.getProperty("directory").toString();
  if (!directory.isEmpty())
  {
    try
    {
      WriteCurrentConfig(directory);
    }
    catch (...)
    {
      MITK_ERROR << "Caught exception while writing info file! Ignoring it and aborting info file.";
    }
  }
  else
  {
    MITK_WARN << "Received igi-recording-started event without directory information! Ignoring it.";
  }
}
