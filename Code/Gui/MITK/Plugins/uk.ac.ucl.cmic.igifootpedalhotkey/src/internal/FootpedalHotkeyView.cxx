/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "FootpedalHotkeyView.h"
#include <ctkDictionary.h>
#include <ctkPluginContext.h>
#include <ctkServiceReference.h>
#include <service/event/ctkEventConstants.h>
#include <service/event/ctkEventAdmin.h>
#include <service/event/ctkEvent.h>
#include "FootpedalHotkeyViewActivator.h"
#include <mitkLogMacros.h>
#include <berryIPreferencesService.h>
#include <berryIBerryPreferences.h>
#include <QDir>
#include <QDateTime>
#include <QmitkWindowsHotkeyHandler.h>


//-----------------------------------------------------------------------------
const char* FootpedalHotkeyView::VIEW_ID = "uk.ac.ucl.cmic.igifootpedalhotkey";


//-----------------------------------------------------------------------------
FootpedalHotkeyView::FootpedalHotkeyView()
  : m_Footswitch1(0)
  , m_IGIRecordingStartedSubscriptionID(-1)
{
}


//-----------------------------------------------------------------------------
FootpedalHotkeyView::~FootpedalHotkeyView()
{
  delete m_Footswitch1;

  // ctk event bus de-registration
  {
    ctkServiceReference ref = mitk::FootpedalHotkeyViewActivator::getContext()->getServiceReference<ctkEventAdmin>();
    if (ref)
    {
      ctkEventAdmin* eventAdmin = mitk::FootpedalHotkeyViewActivator::getContext()->getService<ctkEventAdmin>(ref);
      if (eventAdmin)
      {
        eventAdmin->unsubscribeSlot(m_IGIRecordingStartedSubscriptionID);
        eventAdmin->unpublishSignal(this, SIGNAL(OnStartRecording(ctkDictionary)), "uk/ac/ucl/cmic/IGISTARTRECORDING");
        eventAdmin->unpublishSignal(this, SIGNAL(OnStopRecording(ctkDictionary)),  "uk/ac/ucl/cmic/IGISTOPRECORDING");
      }
    }
  }
}


//-----------------------------------------------------------------------------
std::string FootpedalHotkeyView::GetViewID() const
{
  return VIEW_ID;
}


//-----------------------------------------------------------------------------
void FootpedalHotkeyView::CreateQtPartControl(QWidget* parent)
{
  setupUi(parent);

  ctkServiceReference ref = mitk::FootpedalHotkeyViewActivator::getContext()->getServiceReference<ctkEventAdmin>();
  if (ref)
  {
    ctkEventAdmin* eventAdmin = mitk::FootpedalHotkeyViewActivator::getContext()->getService<ctkEventAdmin>(ref);
    ctkDictionary properties;
    properties[ctkEventConstants::EVENT_TOPIC] = "uk/ac/ucl/cmic/IGIRECORDINGSTARTED";
    m_IGIRecordingStartedSubscriptionID = eventAdmin->subscribeSlot(this, SLOT(OnRecordingStarted(ctkEvent)), properties);

    eventAdmin->publishSignal(this, SIGNAL(OnStartRecording(ctkDictionary)), "uk/ac/ucl/cmic/IGISTARTRECORDING");
    eventAdmin->publishSignal(this, SIGNAL(OnStopRecording(ctkDictionary)),  "uk/ac/ucl/cmic/IGISTOPRECORDING");
  }

  bool ok = false;
  m_Footswitch1 = new QmitkWindowsHotkeyHandler(QmitkWindowsHotkeyHandler::CTRL_ALT_F5);
  ok = QObject::connect(m_Footswitch1, SIGNAL(HotkeyPressed(QmitkWindowsHotkeyHandler*, int)), this, SLOT(OnHotkeyPressed(QmitkWindowsHotkeyHandler*, int)), Qt::QueuedConnection);
  assert(ok);
}


//-----------------------------------------------------------------------------
void FootpedalHotkeyView::OnHotkeyPressed(QmitkWindowsHotkeyHandler* sender, int hotkey)
{
  ctkDictionary   properties;

  switch (hotkey)
  {
    case QmitkWindowsHotkeyHandler::CTRL_ALT_F5:
      emit OnStartRecording(properties);
      break;
  }
}


//-----------------------------------------------------------------------------
void FootpedalHotkeyView::SetFocus()
{
}


//-----------------------------------------------------------------------------
void FootpedalHotkeyView::WriteCurrentConfig(const QString& directory) const
{
  QFile   infoFile(directory + QDir::separator() + VIEW_ID + ".txt");
  bool opened = infoFile.open(QIODevice::ReadWrite | QIODevice::Text | QIODevice::Append);
  if (opened)
  {
    QTextStream   info(&infoFile);
    info.setCodec("UTF-8");
    info << "START: " << QDateTime::currentDateTime().toString() << "\n";
  }
}


//-----------------------------------------------------------------------------
void FootpedalHotkeyView::OnRecordingStarted(const ctkEvent& event)
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
