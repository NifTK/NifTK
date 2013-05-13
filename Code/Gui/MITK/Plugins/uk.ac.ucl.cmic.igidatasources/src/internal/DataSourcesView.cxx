/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

// Qmitk
#include "DataSourcesView.h"
#include <ctkPluginContext.h>
#include <ctkServiceReference.h>
#include <service/event/ctkEventAdmin.h>
#include <service/event/ctkEvent.h>
#include "DataSourcesViewActivator.h"

const std::string DataSourcesView::VIEW_ID = "uk.ac.ucl.cmic.igidatasources";

//-----------------------------------------------------------------------------
DataSourcesView::DataSourcesView()
{
}


//-----------------------------------------------------------------------------
DataSourcesView::~DataSourcesView()
{
}


//-----------------------------------------------------------------------------
std::string DataSourcesView::GetViewID() const
{
  return VIEW_ID;
}


//-----------------------------------------------------------------------------
void DataSourcesView::CreateQtPartControl( QWidget *parent )
{
  m_DataSourceManager = QmitkIGIDataSourceManager::New();
  m_DataSourceManager->setupUi(parent);
  m_DataSourceManager->SetDataStorage(this->GetDataStorage());

  this->RetrievePreferenceValues();

  connect(m_DataSourceManager, SIGNAL(UpdateGuiEnd(igtlUint64)), this, SLOT(OnUpdateGuiEnd(igtlUint64)));

  ctkPluginContext* context = mitk::DataSourcesViewActivator::getContext();
  ctkServiceReference ref = context->getServiceReference<ctkEventAdmin>();
  if (ref)
  {
    ctkEventAdmin* eventAdmin = context->getService<ctkEventAdmin>(ref);
    eventAdmin->publishSignal(this, SIGNAL(Updated(ctkDictionary)),"uk/ac/ucl/cmic/IGIUPDATE");
  }
}


//-----------------------------------------------------------------------------
void DataSourcesView::SetFocus()
{
  m_DataSourceManager->setFocus();
}


//-----------------------------------------------------------------------------
void DataSourcesView::RetrievePreferenceValues()
{
  berry::IPreferences::Pointer prefs = GetPreferences();
  if (prefs.IsNotNull())
  {
    QString path = QString::fromStdString(prefs->Get("output directory prefix", ""));
    if (path == "")
    {
      path = QmitkIGIDataSourceManager::GetDefaultPath();
    }
    QColor errorColour = QmitkIGIDataSourceManager::DEFAULT_ERROR_COLOUR;
    QString errorColourName = QString::fromStdString(prefs->GetByteArray("error colour", ""));
    if (errorColourName != "")
    {
      errorColour = QColor(errorColourName);
    }
    QColor warningColour = QmitkIGIDataSourceManager::DEFAULT_WARNING_COLOUR;
    QString warningColourName = QString::fromStdString(prefs->GetByteArray("warning colour", ""));
    if (warningColourName != "")
    {
      warningColour = QColor(warningColourName);
    }
    QColor okColour = QmitkIGIDataSourceManager::DEFAULT_OK_COLOUR;
    QString okColourName = QString::fromStdString(prefs->GetByteArray("ok colour", ""));
    if (okColourName != "")
    {
      okColour = QColor(okColourName);
    }

    int refreshRate = prefs->GetInt("refresh rate", QmitkIGIDataSourceManager::DEFAULT_FRAME_RATE);
    int clearRate = prefs->GetInt("clear data rate", QmitkIGIDataSourceManager::DEFAULT_CLEAR_RATE);
    int timingTolerance = prefs->GetInt("timing tolerance", QmitkIGIDataSourceManager::DEFAULT_TIMING_TOLERANCE);
    bool saveOnReceipt = prefs->GetBool("save on receive", QmitkIGIDataSourceManager::DEFAULT_SAVE_ON_RECEIPT);
    bool saveInBackground = prefs->GetBool("save in background", QmitkIGIDataSourceManager::DEFAULT_SAVE_IN_BACKGROUND);

    m_DataSourceManager->SetDirectoryPrefix(path);
    m_DataSourceManager->SetFramesPerSecond(refreshRate);
    m_DataSourceManager->SetErrorColour(errorColour);
    m_DataSourceManager->SetWarningColour(warningColour);
    m_DataSourceManager->SetOKColour(okColour);
    m_DataSourceManager->SetClearDataRate(clearRate);
    m_DataSourceManager->SetTimingTolerance(timingTolerance);
    m_DataSourceManager->SetSaveOnReceipt(saveOnReceipt);
    m_DataSourceManager->SetSaveInBackground(saveInBackground);
  }
  else
  {
    QString defaultPath = QmitkIGIDataSourceManager::GetDefaultPath();
    QColor defaultErrorColor = QmitkIGIDataSourceManager::DEFAULT_ERROR_COLOUR;
    QColor defaultWarningColor = QmitkIGIDataSourceManager::DEFAULT_WARNING_COLOUR;
    QColor defaultOKColor = QmitkIGIDataSourceManager::DEFAULT_OK_COLOUR;

    m_DataSourceManager->SetDirectoryPrefix(defaultPath);
    m_DataSourceManager->SetFramesPerSecond(QmitkIGIDataSourceManager::DEFAULT_FRAME_RATE);
    m_DataSourceManager->SetErrorColour(defaultErrorColor);
    m_DataSourceManager->SetWarningColour(defaultWarningColor);
    m_DataSourceManager->SetOKColour(defaultOKColor);
    m_DataSourceManager->SetClearDataRate(QmitkIGIDataSourceManager::DEFAULT_CLEAR_RATE);
    m_DataSourceManager->SetTimingTolerance(QmitkIGIDataSourceManager::DEFAULT_TIMING_TOLERANCE);
    m_DataSourceManager->SetSaveOnReceipt(QmitkIGIDataSourceManager::DEFAULT_SAVE_ON_RECEIPT);
    m_DataSourceManager->SetSaveInBackground(QmitkIGIDataSourceManager::DEFAULT_SAVE_IN_BACKGROUND);
  }
}


//-----------------------------------------------------------------------------
void DataSourcesView::OnPreferencesChanged(const berry::IBerryPreferences*)
{
  this->RetrievePreferenceValues();
}


//-----------------------------------------------------------------------------
void DataSourcesView::OnUpdateGuiEnd(igtlUint64 timeStamp)
{
  ctkDictionary properties;
  properties["timeStamp"] = timeStamp;
  emit Updated(properties);
}
