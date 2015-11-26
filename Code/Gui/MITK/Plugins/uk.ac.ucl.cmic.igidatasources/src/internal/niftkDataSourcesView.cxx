/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkDataSourcesView.h"
#include "niftkDataSourcesViewActivator.h"
#include <ctkPluginContext.h>
#include <ctkServiceReference.h>
#include <service/event/ctkEventAdmin.h>
#include <service/event/ctkEvent.h>
#include <service/event/ctkEventConstants.h>
#include <cassert>

namespace niftk
{

const std::string DataSourcesView::VIEW_ID = "uk.ac.ucl.cmic.igidatasources";

//-----------------------------------------------------------------------------
DataSourcesView::DataSourcesView()
{
}


//-----------------------------------------------------------------------------
DataSourcesView::~DataSourcesView()
{
  ctkPluginContext* context = niftk::DataSourcesViewActivator::getContext();
  if (context)
  {
    ctkServiceReference ref = context->getServiceReference<ctkEventAdmin>();
    if (ref)
    {
      ctkEventAdmin* eventAdmin = context->getService<ctkEventAdmin>(ref);
      if (eventAdmin)
      {
        eventAdmin->unpublishSignal(this, SIGNAL(Updated(ctkDictionary)),"uk/ac/ucl/cmic/IGIUPDATE");
        eventAdmin->unpublishSignal(this, SIGNAL(RecordingStarted(ctkDictionary)), "uk/ac/ucl/cmic/IGIRECORDINGSTARTED");
      }
    }
  }

  bool ok = false;
  ok = QObject::disconnect(m_DataSourceManager, SIGNAL(UpdateGuiFinishedDataSources(niftk::IGIDataType::IGITimeType)), this, SLOT(OnUpdateGuiEnd(niftk::IGIDataType::IGITimeType)));
  assert(ok);
  ok = QObject::disconnect(m_DataSourceManager, SIGNAL(RecordingStarted(QString)), this, SLOT(OnRecordingStarted(QString)));
  assert(ok);
}


//-----------------------------------------------------------------------------
void DataSourcesView::OnRecordingShouldStart(const ctkEvent& event)
{
  m_DataSourceManager->StartRecording();
}


//-----------------------------------------------------------------------------
void DataSourcesView::OnRecordingShouldStop(const ctkEvent& event)
{
  m_DataSourceManager->StopRecording();
}


//-----------------------------------------------------------------------------
void DataSourcesView::OnPreferencesChanged(const berry::IBerryPreferences*)
{
  this->RetrievePreferenceValues();
}


//-----------------------------------------------------------------------------
std::string DataSourcesView::GetViewID() const
{
  return VIEW_ID;
}


//-----------------------------------------------------------------------------
void DataSourcesView::SetFocus()
{
  m_DataSourceManager->setFocus();
}


//-----------------------------------------------------------------------------
void DataSourcesView::CreateQtPartControl( QWidget *parent )
{
  m_DataSourceManager = IGIDataSourceManager::New(this->GetDataStorage());
  m_DataSourceManager->setupUi(parent);

  this->RetrievePreferenceValues();

  bool ok = false;
  ok = QObject::connect(m_DataSourceManager, SIGNAL(UpdateGuiFinishedDataSources(niftk::IGIDataType::IGITimeType)), this, SLOT(OnUpdateGuiEnd(niftk::IGIDataType::IGITimeType)));
  assert(ok);
  ok = QObject::connect(m_DataSourceManager, SIGNAL(RecordingStarted(QString)), this, SLOT(OnRecordingStarted(QString)), Qt::QueuedConnection);
  assert(ok);

  ctkPluginContext* context = niftk::DataSourcesViewActivator::getContext();
  ctkServiceReference ref = context->getServiceReference<ctkEventAdmin>();
  if (ref)
  {
    ctkEventAdmin* eventAdmin = context->getService<ctkEventAdmin>(ref);
    eventAdmin->publishSignal(this, SIGNAL(Updated(ctkDictionary)),"uk/ac/ucl/cmic/IGIUPDATE");
    eventAdmin->publishSignal(this, SIGNAL(RecordingStarted(ctkDictionary)), "uk/ac/ucl/cmic/IGIRECORDINGSTARTED");

    ctkDictionary properties;
    properties[ctkEventConstants::EVENT_TOPIC] = "uk/ac/ucl/cmic/IGISTARTRECORDING";
    eventAdmin->subscribeSlot(this, SLOT(OnRecordingShouldStart(ctkEvent)), properties);
    properties[ctkEventConstants::EVENT_TOPIC] = "uk/ac/ucl/cmic/IGISTOPRECORDING";
    eventAdmin->subscribeSlot(this, SLOT(OnRecordingShouldStop(ctkEvent)), properties);
  }
}


//-----------------------------------------------------------------------------
void DataSourcesView::RetrievePreferenceValues()
{
  berry::IPreferences::Pointer prefs = GetPreferences();
  if (prefs.IsNotNull())
  {
    QString path = prefs->Get("output directory prefix", "");
    if (path == "")
    {
      path = niftk::IGIDataSourceManager::GetDefaultPath();
    }

    int refreshRate = prefs->GetInt("refresh rate", niftk::IGIDataSourceManager::DEFAULT_FRAME_RATE);

    m_DataSourceManager->SetDirectoryPrefix(path);
    m_DataSourceManager->SetFramesPerSecond(refreshRate);
  }
  else
  {
    QString defaultPath = niftk::IGIDataSourceManager::GetDefaultPath();
    m_DataSourceManager->SetDirectoryPrefix(defaultPath);
    m_DataSourceManager->SetFramesPerSecond(niftk::IGIDataSourceManager::DEFAULT_FRAME_RATE);
  }
}


//-----------------------------------------------------------------------------
void DataSourcesView::OnUpdateGuiEnd(niftk::IGIDataType::IGITimeType timeStamp)
{
  ctkDictionary properties;
  properties["timeStamp"] = timeStamp;
  emit Updated(properties);
}


//-----------------------------------------------------------------------------
void DataSourcesView::OnRecordingStarted(QString baseDirectory)
{
  ctkDictionary properties;
  properties["directory"] = baseDirectory;
  emit RecordingStarted(properties);
}

} // end namespace
