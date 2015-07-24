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

const QString XnatPluginPreferencePage::SERVER_NAME("Server");
const QString XnatPluginPreferencePage::SERVER_DEFAULT("https://central.xnat.org");
const QString XnatPluginPreferencePage::USER_NAME("User");
const QString XnatPluginPreferencePage::USER_DEFAULT("guest");
const QString XnatPluginPreferencePage::DOWNLOAD_DIRECTORY_NAME("Download directory");
const QString XnatPluginPreferencePage::DOWNLOAD_DIRECTORY_DEFAULT(".");
const QString XnatPluginPreferencePage::WORK_DIRECTORY_NAME("Work directory");
const QString XnatPluginPreferencePage::WORK_DIRECTORY_DEFAULT(".");

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
  berry::IPreferencesService* prefService = berry::Platform::GetPreferencesService();

  QString browserViewPreferencesName = "/";
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

  m_XnatBrowserViewPreferences->Put(SERVER_NAME, server);
  m_XnatBrowserViewPreferences->Put(USER_NAME, user);
  m_XnatBrowserViewPreferences->Put(DOWNLOAD_DIRECTORY_NAME, downloadDirectory);
  m_XnatBrowserViewPreferences->Put(WORK_DIRECTORY_NAME, workDirectory);

  return true;
}

void XnatPluginPreferencePage::PerformCancel()
{
}

void XnatPluginPreferencePage::Update()
{
  QString server = m_XnatBrowserViewPreferences->Get(SERVER_NAME, SERVER_DEFAULT);
  QString user = m_XnatBrowserViewPreferences->Get(USER_NAME, USER_DEFAULT);
  QString downloadDirectory = m_XnatBrowserViewPreferences->Get(DOWNLOAD_DIRECTORY_NAME, DOWNLOAD_DIRECTORY_DEFAULT);
  QString workDirectory = m_XnatBrowserViewPreferences->Get(WORK_DIRECTORY_NAME, WORK_DIRECTORY_DEFAULT);

  m_Controls->ldtServer->setText(server);
  m_Controls->ldtUser->setText(user);
  m_Controls->dirBtnDownloadDirectory->setDirectory(downloadDirectory);
  m_Controls->dirBtnWorkDirectory->setDirectory(workDirectory);
}
