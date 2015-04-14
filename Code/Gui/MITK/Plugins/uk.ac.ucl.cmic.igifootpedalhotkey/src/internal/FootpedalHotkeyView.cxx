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


//-----------------------------------------------------------------------------
const char* FootpedalHotkeyView::VIEW_ID = "uk.ac.ucl.cmic.igifootpedalhotkey";


//-----------------------------------------------------------------------------
FootpedalHotkeyView::FootpedalHotkeyView()
  : m_IGIRecordingStartedSubscriptionID(-1)
{
}


//-----------------------------------------------------------------------------
FootpedalHotkeyView::~FootpedalHotkeyView()
{
  // ctk event bus de-registration
  {
    ctkServiceReference ref = mitk::FootpedalHotkeyViewActivator::getContext()->getServiceReference<ctkEventAdmin>();
    if (ref)
    {
      ctkEventAdmin* eventAdmin = mitk::FootpedalHotkeyViewActivator::getContext()->getService<ctkEventAdmin>(ref);
      if (eventAdmin)
      {
        eventAdmin->unsubscribeSlot(m_IGIRecordingStartedSubscriptionID);
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
