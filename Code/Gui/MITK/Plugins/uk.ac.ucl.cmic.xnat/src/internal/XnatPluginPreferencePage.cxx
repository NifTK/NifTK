/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "XnatPluginPreferencePage.h"

#include <QWidget>

#include <berryIPreferencesService.h>
#include <berryPlatform.h>

#include "XnatBrowserView.h"

const std::string XnatPluginPreferencePage::SERVER_NAME("Server");
const std::string XnatPluginPreferencePage::SERVER_DEFAULT("https://central.xnat.org");
const std::string XnatPluginPreferencePage::USER_NAME("User");
const std::string XnatPluginPreferencePage::USER_DEFAULT("guest");
const std::string XnatPluginPreferencePage::DOWNLOAD_DIRECTORY_NAME("Download directory");
const std::string XnatPluginPreferencePage::DOWNLOAD_DIRECTORY_DEFAULT(".");
const std::string XnatPluginPreferencePage::WORK_DIRECTORY_NAME("Work directory");
const std::string XnatPluginPreferencePage::WORK_DIRECTORY_DEFAULT(".");

XnatPluginPreferencePage::XnatPluginPreferencePage()
: m_Initializing(false)
, m_MainControl(0)
, m_Controls(0)
{
}

XnatPluginPreferencePage::~XnatPluginPreferencePage()
{
  if (m_Controls)
  {
    delete m_MainControl;
    delete m_Controls;
  }
}

void XnatPluginPreferencePage::Init(berry::IWorkbench::Pointer )
{

}

void XnatPluginPreferencePage::CreateQtControl(QWidget* parent)
{
  m_Initializing = true;
  berry::IPreferencesService::Pointer prefService =
      berry::Platform::GetServiceRegistry().GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);

  std::string browserViewPreferencesName = "/";
  browserViewPreferencesName += XnatBrowserView::VIEW_ID;
  m_XnatBrowserViewPreferences = prefService->GetSystemPreferences()->Node(browserViewPreferencesName);

  if (!m_Controls)
  {
    m_MainControl = new QWidget(parent);

    // Create UI
    m_Controls = new Ui::XnatPluginPreferencePage();
    m_Controls->setupUi(m_MainControl);
  }

  this->Update();

  m_Initializing = false;
}

QWidget* XnatPluginPreferencePage::GetQtControl() const
{
  return m_MainControl;
}

bool XnatPluginPreferencePage::PerformOk()
{
  QString server = m_Controls->ldtServer->text();
  QString user = m_Controls->ldtUser->text();
  QString downloadDirectory = m_Controls->dirBtnDownloadDirectory->directory();
  QString workDirectory = m_Controls->dirBtnWorkDirectory->directory();

  m_XnatBrowserViewPreferences->Put(SERVER_NAME, server.toStdString());
  m_XnatBrowserViewPreferences->Put(USER_NAME, user.toStdString());
  m_XnatBrowserViewPreferences->Put(DOWNLOAD_DIRECTORY_NAME, downloadDirectory.toStdString());
  m_XnatBrowserViewPreferences->Put(WORK_DIRECTORY_NAME, workDirectory.toStdString());

  return true;
}

void XnatPluginPreferencePage::PerformCancel()
{
}

void XnatPluginPreferencePage::Update()
{
  std::string server = m_XnatBrowserViewPreferences->Get(SERVER_NAME, SERVER_DEFAULT);
  std::string user = m_XnatBrowserViewPreferences->Get(USER_NAME, USER_DEFAULT);
  std::string downloadDirectory = m_XnatBrowserViewPreferences->Get(DOWNLOAD_DIRECTORY_NAME, DOWNLOAD_DIRECTORY_DEFAULT);
  std::string workDirectory = m_XnatBrowserViewPreferences->Get(WORK_DIRECTORY_NAME, WORK_DIRECTORY_DEFAULT);

  m_Controls->ldtServer->setText(QString::fromStdString(server));
  m_Controls->ldtUser->setText(QString::fromStdString(user));
  m_Controls->dirBtnDownloadDirectory->setDirectory(QString::fromStdString(downloadDirectory));
  m_Controls->dirBtnWorkDirectory->setDirectory(QString::fromStdString(workDirectory));
}
