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
#include <niftkWindowsHotkeyHandler.h>


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
        eventAdmin->unpublishSignal(this, SIGNAL(OnFootSwitch1Start(ctkDictionary)), "uk/ac/ucl/cmic/IGIFOOTSWITCH1START");
        eventAdmin->unpublishSignal(this, SIGNAL(OnFootSwitch1Stop(ctkDictionary)),  "uk/ac/ucl/cmic/IGIFOOTSWITCH1STOP");
        eventAdmin->unpublishSignal(this, SIGNAL(OnFootSwitch2Start(ctkDictionary)), "uk/ac/ucl/cmic/IGIFOOTSWITCH2START");
        eventAdmin->unpublishSignal(this, SIGNAL(OnFootSwitch2Stop(ctkDictionary)),  "uk/ac/ucl/cmic/IGIFOOTSWITCH2STOP");
        eventAdmin->unpublishSignal(this, SIGNAL(OnFootSwitch3Start(ctkDictionary)), "uk/ac/ucl/cmic/IGIFOOTSWITCH3START");
        eventAdmin->unpublishSignal(this, SIGNAL(OnFootSwitch3Stop(ctkDictionary)),  "uk/ac/ucl/cmic/IGIFOOTSWITCH3STOP");
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

    eventAdmin->publishSignal(this, SIGNAL(OnFootSwitch1Start(ctkDictionary)), "uk/ac/ucl/cmic/IGIFOOTSWITCH1START");
    eventAdmin->publishSignal(this, SIGNAL(OnFootSwitch1Stop(ctkDictionary)),  "uk/ac/ucl/cmic/IGIFOOTSWITCH1STOP");
    eventAdmin->publishSignal(this, SIGNAL(OnFootSwitch2Start(ctkDictionary)), "uk/ac/ucl/cmic/IGIFOOTSWITCH2START");
    eventAdmin->publishSignal(this, SIGNAL(OnFootSwitch2Stop(ctkDictionary)),  "uk/ac/ucl/cmic/IGIFOOTSWITCH2STOP");
    eventAdmin->publishSignal(this, SIGNAL(OnFootSwitch3Start(ctkDictionary)), "uk/ac/ucl/cmic/IGIFOOTSWITCH3START");
    eventAdmin->publishSignal(this, SIGNAL(OnFootSwitch3Stop(ctkDictionary)),  "uk/ac/ucl/cmic/IGIFOOTSWITCH3STOP");
  }

  bool ok = false;
  m_Footswitch1 = new niftk::WindowsHotkeyHandler(niftk::WindowsHotkeyHandler::CTRL_ALT_F5);
  ok = QObject::connect(m_Footswitch1, SIGNAL(HotkeyPressed(niftk::WindowsHotkeyHandler*, int)), this, SLOT(OnHotkeyPressed(niftk::WindowsHotkeyHandler*, int)), Qt::QueuedConnection);
  assert(ok);
  m_Footswitch2 = new niftk::WindowsHotkeyHandler(niftk::WindowsHotkeyHandler::CTRL_ALT_F6);
  ok = QObject::connect(m_Footswitch2, SIGNAL(HotkeyPressed(niftk::WindowsHotkeyHandler*, int)), this, SLOT(OnHotkeyPressed(niftk::WindowsHotkeyHandler*, int)), Qt::QueuedConnection);
  assert(ok);
  m_Footswitch3 = new niftk::WindowsHotkeyHandler(niftk::WindowsHotkeyHandler::CTRL_ALT_F7);
  ok = QObject::connect(m_Footswitch3, SIGNAL(HotkeyPressed(niftk::WindowsHotkeyHandler*, int)), this, SLOT(OnHotkeyPressed(niftk::WindowsHotkeyHandler*, int)), Qt::QueuedConnection);
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
  MITK_INFO << "Stopping footpedal/hotkey 1.";
  ctkDictionary   properties;
  emit OnFootSwitch1Stop(properties);

  m_LeftLabel->setText("");
}


//-----------------------------------------------------------------------------
void FootpedalHotkeyView::OnTimer2()
{
  MITK_INFO << "Stopping footpedal/hotkey 2.";
  ctkDictionary   properties;
  emit OnFootSwitch2Stop(properties);

  m_MiddleLabel->setText("");
}


//-----------------------------------------------------------------------------
void FootpedalHotkeyView::OnTimer3()
{
  MITK_INFO << "Stopping footpedal/hotkey 3.";
  ctkDictionary   properties;
  emit OnFootSwitch3Stop(properties);

  m_RightLabel->setText("");
}


//-----------------------------------------------------------------------------
void FootpedalHotkeyView::OnHotkeyPressed(niftk::WindowsHotkeyHandler* sender, int hotkey)
{
  ctkDictionary   properties;

  switch (hotkey)
  {
    case niftk::WindowsHotkeyHandler::CTRL_ALT_F5:
      if (!m_Footswitch1OffTimer->isActive())
      {
        MITK_INFO << "Starting footpedal/hotkey 1.";
        m_LeftLabel->setText("X");
        emit OnFootSwitch1Start(properties);
      }

      // if we get another hotkey event shortly, i.e. user is still pressing the key,
      // and the system generates key-repeat events, then reset timer.
      // otherwise it will expire at some point, and signal a hotkey-release.
      m_Footswitch1OffTimer->start();
      break;

    case niftk::WindowsHotkeyHandler::CTRL_ALT_F6:
      if (!m_Footswitch2OffTimer->isActive())
      {
        MITK_INFO << "Starting footpedal/hotkey 2.";
        m_MiddleLabel->setText("X");
        emit OnFootSwitch2Start(properties);
      }
      m_Footswitch2OffTimer->start();
      break;

    case niftk::WindowsHotkeyHandler::CTRL_ALT_F7:
      if (!m_Footswitch3OffTimer->isActive())
      {
        MITK_INFO << "Starting footpedal/hotkey 3.";
        m_RightLabel->setText("X");
        emit OnFootSwitch3Start(properties);
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
