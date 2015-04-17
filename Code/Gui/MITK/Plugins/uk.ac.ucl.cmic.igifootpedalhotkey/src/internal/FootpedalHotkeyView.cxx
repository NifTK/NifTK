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
  , m_Footswitch1OffTimer(0)
  , m_Footswitch2(0)
  , m_Footswitch2OffTimer(0)
  , m_Footswitch3(0)
  , m_Footswitch3OffTimer(0)
  , m_IGIRecordingStartedSubscriptionID(-1)
{
}


//-----------------------------------------------------------------------------
FootpedalHotkeyView::~FootpedalHotkeyView()
{
  delete m_Footswitch1;
  delete m_Footswitch1OffTimer;
  delete m_Footswitch2;
  delete m_Footswitch2OffTimer;
  delete m_Footswitch3;
  delete m_Footswitch3OffTimer;

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
  m_LeftLabel->setText("");
  m_MiddleLabel->setText("");
  m_RightLabel->setText("");

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
  m_Footswitch2 = new QmitkWindowsHotkeyHandler(QmitkWindowsHotkeyHandler::CTRL_ALT_F6);
  ok = QObject::connect(m_Footswitch2, SIGNAL(HotkeyPressed(QmitkWindowsHotkeyHandler*, int)), this, SLOT(OnHotkeyPressed(QmitkWindowsHotkeyHandler*, int)), Qt::QueuedConnection);
  assert(ok);
  m_Footswitch3 = new QmitkWindowsHotkeyHandler(QmitkWindowsHotkeyHandler::CTRL_ALT_F7);
  ok = QObject::connect(m_Footswitch3, SIGNAL(HotkeyPressed(QmitkWindowsHotkeyHandler*, int)), this, SLOT(OnHotkeyPressed(QmitkWindowsHotkeyHandler*, int)), Qt::QueuedConnection);
  assert(ok);

  m_Footswitch1OffTimer = new QTimer(this);
  m_Footswitch1OffTimer->setSingleShot(true);
  m_Footswitch1OffTimer->setInterval(1000);       // should be slightly longer than key-repeat!
  ok = QObject::connect(m_Footswitch1OffTimer, SIGNAL(timeout()), this, SLOT(OnTimer1()));
  assert(ok);

  m_Footswitch2OffTimer = new QTimer(this);
  m_Footswitch2OffTimer->setSingleShot(true);
  m_Footswitch2OffTimer->setInterval(1000);       // should be slightly longer than key-repeat!
  ok = QObject::connect(m_Footswitch2OffTimer, SIGNAL(timeout()), this, SLOT(OnTimer2()));
  assert(ok);

  m_Footswitch3OffTimer = new QTimer(this);
  m_Footswitch3OffTimer->setSingleShot(true);
  m_Footswitch3OffTimer->setInterval(1000);       // should be slightly longer than key-repeat!
  ok = QObject::connect(m_Footswitch3OffTimer, SIGNAL(timeout()), this, SLOT(OnTimer3()));
  assert(ok);
}


//-----------------------------------------------------------------------------
void FootpedalHotkeyView::OnTimer1()
{
  MITK_INFO << "Stopping recording due to footpedal/hotkey 1";
  ctkDictionary   properties;
  emit OnStopRecording(properties);

  m_LeftLabel->setText("");
}


//-----------------------------------------------------------------------------
void FootpedalHotkeyView::OnTimer2()
{
  // what action?
  MITK_INFO << "Unassigned footpedal 2 release.";

  m_MiddleLabel->setText("");
}


//-----------------------------------------------------------------------------
void FootpedalHotkeyView::OnTimer3()
{
  // what action?
  MITK_INFO << "Unassigned footpedal 3 release.";

  m_RightLabel->setText("");
}


//-----------------------------------------------------------------------------
void FootpedalHotkeyView::OnHotkeyPressed(QmitkWindowsHotkeyHandler* sender, int hotkey)
{
  ctkDictionary   properties;

  switch (hotkey)
  {
    case QmitkWindowsHotkeyHandler::CTRL_ALT_F5:
      if (!m_Footswitch1OffTimer->isActive())
      {
        MITK_INFO << "Starting recording due to footpedal/hotkey 1.";
        m_LeftLabel->setText("X");
        emit OnStartRecording(properties);
      }

      // if we get another hotkey event shortly, i.e. user is still pressing the key,
      // and the system generates key-repeat events, then reset timer.
      // otherwise it will expire at some point, and signal a hotkey-release.
      m_Footswitch1OffTimer->start();
      break;

    case QmitkWindowsHotkeyHandler::CTRL_ALT_F6:
      if (!m_Footswitch2OffTimer->isActive())
      {
        MITK_INFO << "Unassigned footpedal 2 press.";
        m_MiddleLabel->setText("X");
        // what action?
      }
      m_Footswitch2OffTimer->start();
      break;

    case QmitkWindowsHotkeyHandler::CTRL_ALT_F7:
      if (!m_Footswitch3OffTimer->isActive())
      {
        MITK_INFO << "Unassigned footpedal 3 press.";
        m_RightLabel->setText("X");
        // what action?
      }
      m_Footswitch3OffTimer->start();
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
