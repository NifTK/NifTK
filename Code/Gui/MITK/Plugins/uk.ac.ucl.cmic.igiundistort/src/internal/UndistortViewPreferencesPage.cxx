/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "UndistortViewPreferencesPage.h"

#include <QFormLayout>
#include <QVBoxLayout>
#include <QLabel>
#include <QCheckBox>
#include <QMessageBox>
#include <QFileDialog>
#include <QSpinBox>
#include "ui_UndistortViewPreferencePage.h"
#include <berryIPreferencesService.h>
#include <berryPlatform.h>


//-----------------------------------------------------------------------------
const char* UndistortViewPreferencesPage::s_PrefsNodeName                         = "/uk.ac.ucl.cmic.igiundistort";
const char* UndistortViewPreferencesPage::s_DefaultCalibrationFilePathPrefsName   = "default calib file path";


//-----------------------------------------------------------------------------
UndistortViewPreferencesPage::UndistortViewPreferencesPage()
{

}


//-----------------------------------------------------------------------------
UndistortViewPreferencesPage::UndistortViewPreferencesPage(const UndistortViewPreferencesPage& other)
: berry::Object(), QObject()
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
UndistortViewPreferencesPage::~UndistortViewPreferencesPage()
{

}


//-----------------------------------------------------------------------------
void UndistortViewPreferencesPage::Init(berry::IWorkbench::Pointer )
{
}


//-----------------------------------------------------------------------------
void UndistortViewPreferencesPage::OnDefaultPathBrowseButtonClicked()
{
  QString   file = QFileDialog::getExistingDirectory(GetQtControl(), "Calibration File Path", m_DefaultCalibrationFilePath);
  if (!file.isEmpty())
  {
    m_DefaultCalibrationFilePath = file;
    m_DefaultFilePathLineEdit->setText(m_DefaultCalibrationFilePath);
  }
}


//-----------------------------------------------------------------------------
void UndistortViewPreferencesPage::CreateQtControl(QWidget* parent)
{
  setupUi(parent);
  connect(m_DefaultFilePathBrowseButton, SIGNAL(clicked()), this, SLOT(OnDefaultPathBrowseButtonClicked()));

  berry::IPreferencesService::Pointer prefService = berry::Platform::GetServiceRegistry().GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);
  m_UndistortPreferencesNode = prefService->GetSystemPreferences()->Node(s_PrefsNodeName);

  // read prefs and stuff it into controls
  Update();
}


//-----------------------------------------------------------------------------
QWidget* UndistortViewPreferencesPage::GetQtControl() const
{
  return UndistortViewPreferencePageWidget;
}


//-----------------------------------------------------------------------------
bool UndistortViewPreferencesPage::PerformOk()
{
  m_UndistortPreferencesNode->Put(s_DefaultCalibrationFilePathPrefsName, m_DefaultFilePathLineEdit->text().toStdString());

  return true;
}


//-----------------------------------------------------------------------------
void UndistortViewPreferencesPage::PerformCancel()
{

}


//-----------------------------------------------------------------------------
void UndistortViewPreferencesPage::Update()
{
  m_DefaultCalibrationFilePath = QString::fromStdString(m_UndistortPreferencesNode->Get(s_DefaultCalibrationFilePathPrefsName, ""));
  m_DefaultFilePathLineEdit->setText(m_DefaultCalibrationFilePath);
}
