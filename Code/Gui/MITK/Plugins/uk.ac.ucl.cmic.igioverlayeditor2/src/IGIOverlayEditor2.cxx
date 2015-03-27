/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "IGIOverlayEditor2.h"

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

#include <mitkColorProperty.h>
#include <mitkGlobalInteraction.h>
#include <mitkNodePredicateNot.h>
#include <mitkNodePredicateProperty.h>

#include <mitkDataStorageEditorInput.h>
#include <mitkIDataStorageService.h>

#include <OverlayEditor2/QmitkIGIOverlayEditor2.h>
#include <internal/IGIOverlayEditor2PreferencePage.h>
#include <internal/IGIOverlayEditor2Activator.h>


//-----------------------------------------------------------------------------
const char* IGIOverlayEditor2::EDITOR_ID = "org.mitk.editors.igioverlayeditor2";


/**
 * \class IGIOverlayEditor2Private
 * \brief PIMPL pattern implementation of IGIOverlayEditor2.
 */
class IGIOverlayEditor2Private
{
public:

  IGIOverlayEditor2Private();
  ~IGIOverlayEditor2Private();

  QmitkIGIOverlayEditor2* m_IGIOverlayEditor2;
  //std::string m_FirstBackgroundColor;
  //std::string m_SecondBackgroundColor;
  berry::IPartListener::Pointer m_PartListener;
};


/**
 * \class IGIOverlayWidgetPartListener
 * \brief Used to handle interaction with the contained overlay
 * editor widget when this IGIOverlayEditor is opened/closed etc.
 */
struct IGIOverlay2WidgetPartListener : public berry::IPartListener
{
  berryObjectMacro(IGIOverlay2WidgetPartListener)

  //---------------------------------------------------------------------------
  IGIOverlay2WidgetPartListener(IGIOverlayEditor2Private* dd)
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
    if (partRef->GetId() == IGIOverlayEditor2::EDITOR_ID)
    {
      IGIOverlayEditor2::Pointer editor = partRef->GetPart(false).Cast<IGIOverlayEditor2>();
      if (d->m_IGIOverlayEditor2 == editor->GetIGIOverlayEditor2())
      {
        // Call editor to turn things off as the widget is being closed.
      }
    }
  }

  //---------------------------------------------------------------------------
  void PartHidden(berry::IWorkbenchPartReference::Pointer partRef)
  {
    if (partRef->GetId() == IGIOverlayEditor2::EDITOR_ID)
    {
      IGIOverlayEditor2::Pointer editor = partRef->GetPart(false).Cast<IGIOverlayEditor2>();
      if (d->m_IGIOverlayEditor2 == editor->GetIGIOverlayEditor2())
      {
        // Call editor to turn things off as the widget is being hidden.
      }
    }
  }

  //---------------------------------------------------------------------------
  void PartVisible(berry::IWorkbenchPartReference::Pointer partRef)
  {
    if (partRef->GetId() == IGIOverlayEditor2::EDITOR_ID)
    {
      IGIOverlayEditor2::Pointer editor = partRef->GetPart(false).Cast<IGIOverlayEditor2>();
      if (d->m_IGIOverlayEditor2 == editor->GetIGIOverlayEditor2())
      {
        // Call editor to turn things on as the widget is being made visible.
      }
    }
  }

private:

  IGIOverlayEditor2Private* const d;
};


//-----------------------------------------------------------------------------
IGIOverlayEditor2Private::IGIOverlayEditor2Private()
  : m_IGIOverlayEditor2(0)
  , m_PartListener(new IGIOverlay2WidgetPartListener(this))
{
}


//-----------------------------------------------------------------------------
IGIOverlayEditor2Private::~IGIOverlayEditor2Private()
{
}

//-----------------------------------------------------------------------------
IGIOverlayEditor2::IGIOverlayEditor2()
  : d(new IGIOverlayEditor2Private)
{
}


//-----------------------------------------------------------------------------
IGIOverlayEditor2::~IGIOverlayEditor2()
{
  this->GetSite()->GetPage()->RemovePartListener(d->m_PartListener);
}


//-----------------------------------------------------------------------------
QmitkIGIOverlayEditor2* IGIOverlayEditor2::GetIGIOverlayEditor2()
{
  return d->m_IGIOverlayEditor2;
}


//-----------------------------------------------------------------------------
QmitkRenderWindow *IGIOverlayEditor2::GetActiveQmitkRenderWindow() const
{
  return 0;
}


//-----------------------------------------------------------------------------
QHash<QString, QmitkRenderWindow *> IGIOverlayEditor2::GetQmitkRenderWindows() const
{
  return QHash<QString, QmitkRenderWindow *>();
}


//-----------------------------------------------------------------------------
QmitkRenderWindow *IGIOverlayEditor2::GetQmitkRenderWindow(const QString &id) const
{
  return 0;
}


//-----------------------------------------------------------------------------
mitk::Point3D IGIOverlayEditor2::GetSelectedPosition(const QString & id) const
{
  // Not implemented.
  mitk::Point3D point;
  point[0] = 0;
  point[1] = 0;
  point[2] = 0;
  return point;
}


//-----------------------------------------------------------------------------
void IGIOverlayEditor2::SetSelectedPosition(const mitk::Point3D &pos, const QString &id)
{
  // Not implemented.
}


//-----------------------------------------------------------------------------
void IGIOverlayEditor2::EnableDecorations(bool /*enable*/, const QStringList & /*decorations*/)
{
}


//-----------------------------------------------------------------------------
bool IGIOverlayEditor2::IsDecorationEnabled(const QString & /*decoration*/) const
{
  return false;
}


//-----------------------------------------------------------------------------
QStringList IGIOverlayEditor2::GetDecorations() const
{
  QStringList decorations;
  return decorations;
}


//-----------------------------------------------------------------------------
mitk::SlicesRotator* IGIOverlayEditor2::GetSlicesRotator() const
{
  return NULL;
}


//-----------------------------------------------------------------------------
mitk::SlicesSwiveller* IGIOverlayEditor2::GetSlicesSwiveller() const
{
  return NULL;
}


//-----------------------------------------------------------------------------
void IGIOverlayEditor2::EnableSlicingPlanes(bool /*enable*/)
{
}


//-----------------------------------------------------------------------------
bool IGIOverlayEditor2::IsSlicingPlanesEnabled() const
{
  return false;
}


//-----------------------------------------------------------------------------
void IGIOverlayEditor2::EnableLinkedNavigation(bool /*enable*/)
{
}


//-----------------------------------------------------------------------------
bool IGIOverlayEditor2::IsLinkedNavigationEnabled() const
{
  return false;
}


//-----------------------------------------------------------------------------
void IGIOverlayEditor2::CreateQtPartControl(QWidget* parent)
{
  if (d->m_IGIOverlayEditor2 == 0)
  {
    QHBoxLayout* layout = new QHBoxLayout(parent);
    layout->setContentsMargins(0,0,0,0);

    d->m_IGIOverlayEditor2 = new QmitkIGIOverlayEditor2(parent);
    layout->addWidget(d->m_IGIOverlayEditor2);


    ctkPluginContext*     context     = mitk::IGIOverlayEditor2Activator::/*GetDefault()->*/getContext();
    ctkServiceReference   serviceRef  = context->getServiceReference<OclResourceService>();
    OclResourceService*   oclService  = context->getService<OclResourceService>(serviceRef);
    if (oclService == NULL)
    {
      mitkThrow() << "Failed to find OpenCL resource service." << std::endl;
    }
    d->m_IGIOverlayEditor2->SetOclResourceService(oclService);


    mitk::DataStorage::Pointer ds = this->GetDataStorage();
    d->m_IGIOverlayEditor2->SetDataStorage(ds);

    this->GetSite()->GetPage()->AddPartListener(d->m_PartListener);

    QMetaObject::invokeMethod(this, "OnPreferencesChanged", Qt::QueuedConnection);

    this->RequestUpdate();

    // Finally: Listen to update pulse coming off of event bus. This pulse comes from the data manager updating.
    ctkServiceReference ref = mitk::IGIOverlayEditor2Activator::getContext()->getServiceReference<ctkEventAdmin>();
    if (ref)
    {
      ctkEventAdmin* eventAdmin = mitk::IGIOverlayEditor2Activator::getContext()->getService<ctkEventAdmin>(ref);
      
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
void IGIOverlayEditor2::OnPreferencesChanged()
{
  berry::IPreferencesService::Pointer prefService = berry::Platform::GetServiceRegistry().GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);
  berry::IBerryPreferences::Pointer   prefsNode   = prefService->GetSystemPreferences()->Node(EDITOR_ID).Cast<berry::IBerryPreferences>();

  this->OnPreferencesChanged(prefsNode.GetPointer());
}


//-----------------------------------------------------------------------------
void IGIOverlayEditor2::OnPreferencesChanged(const berry::IBerryPreferences* prefs)
{
  // 0xAABBGGRR
  unsigned int   backgroundColour = prefs->GetInt(IGIOverlayEditor2PreferencePage::BACKGROUND_COLOR_PREFSKEY, 0x00000000);

  if (d->m_IGIOverlayEditor2 != 0)
  {
    d->m_IGIOverlayEditor2->SetBackgroundColour(0xFF00FFFF);
  }
}


//-----------------------------------------------------------------------------
void IGIOverlayEditor2::SetFocus()
{
  if (d->m_IGIOverlayEditor2 != 0)
  {
    d->m_IGIOverlayEditor2->setFocus();
  }
}


//-----------------------------------------------------------------------------
void IGIOverlayEditor2::OnIGIUpdate(const ctkEvent& event)
{
  d->m_IGIOverlayEditor2->Update();
}


//-----------------------------------------------------------------------------
void IGIOverlayEditor2::WriteCurrentConfig(const QString& directory) const
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
void IGIOverlayEditor2::OnRecordingStarted(const ctkEvent& event)
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
