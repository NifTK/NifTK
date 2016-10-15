/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkDataSourcesViewPreferencePage.h"
#include "niftkDataSourcesView.h"
#include <niftkIGIDataSourceManager.h>

#include <QFormLayout>
#include <QVBoxLayout>
#include <QLabel>
#include <QWidget>
#include <QDir>
#include <QSpinBox>
#include <ctkDirectoryButton.h>
#include <berryIPreferencesService.h>
#include <berryPlatform.h>

namespace niftk
{

//-----------------------------------------------------------------------------
DataSourcesViewPreferencePage::DataSourcesViewPreferencePage()
: m_MainControl(0)
, m_FramesPerSecondSpinBox(0)
, m_DirectoryPrefix(0)
, m_Initializing(false)
, m_DataSourcesViewPreferencesNode(0)
{
}


//-----------------------------------------------------------------------------
DataSourcesViewPreferencePage::DataSourcesViewPreferencePage(const DataSourcesViewPreferencePage& other)
: berry::Object(), QObject()
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
DataSourcesViewPreferencePage::~DataSourcesViewPreferencePage()
{
}


//-----------------------------------------------------------------------------
void DataSourcesViewPreferencePage::Init(berry::IWorkbench::Pointer )
{
  // no-op.
}


//-----------------------------------------------------------------------------
void DataSourcesViewPreferencePage::CreateQtControl(QWidget* parent)
{
  m_Initializing = true;

  berry::IPreferencesService* prefService = berry::Platform::GetPreferencesService();
  m_DataSourcesViewPreferencesNode = prefService->GetSystemPreferences()->Node(niftk::DataSourcesView::VIEW_ID);

  m_MainControl = new QWidget(parent);
  QFormLayout *formLayout = new QFormLayout;

  m_FramesPerSecondSpinBox = new QSpinBox();
  m_FramesPerSecondSpinBox->setMinimum(1);
  m_FramesPerSecondSpinBox->setMaximum(50);
  m_DirectoryPrefix = new ctkDirectoryButton();

  formLayout->addRow("refresh rate (per second)", m_FramesPerSecondSpinBox);
  formLayout->addRow("output directory prefix", m_DirectoryPrefix);

  m_MainControl->setLayout(formLayout);
  this->Update();

  m_Initializing = false;
}


//-----------------------------------------------------------------------------
QWidget* DataSourcesViewPreferencePage::GetQtControl() const
{
  return m_MainControl;
}


//-----------------------------------------------------------------------------
bool DataSourcesViewPreferencePage::PerformOk()
{
  m_DataSourcesViewPreferencesNode->PutInt("refresh rate", m_FramesPerSecondSpinBox->value());
  m_DataSourcesViewPreferencesNode->Put("output directory prefix", m_DirectoryPrefix->directory());
  return true;
}


//-----------------------------------------------------------------------------
void DataSourcesViewPreferencePage::PerformCancel()
{
  // no-op.
}


//-----------------------------------------------------------------------------
void DataSourcesViewPreferencePage::Update()
{
  m_FramesPerSecondSpinBox->setValue(m_DataSourcesViewPreferencesNode->GetInt("refresh rate", niftk::IGIDataSourceManager::DEFAULT_FRAME_RATE));
  QString path = m_DataSourcesViewPreferencesNode->Get("output directory prefix", "");
  if (path == "")
  {
    path = niftk::IGIDataSourceManager::GetDefaultPath();
  }
  m_DirectoryPrefix->setDirectory(path);
}

} // end namespace
