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
#include <QSpinBox>

#include <berryIPreferencesService.h>
#include <berryPlatform.h>

#if 0
const std::string ImageStatisticsViewPreferencesPage::AUTO_UPDATE_NAME("auto update");
const std::string ImageStatisticsViewPreferencesPage::ASSUME_BINARY_NAME("assume binary");
const std::string ImageStatisticsViewPreferencesPage::REQUIRE_SAME_SIZE_IMAGE_NAME("require same size image");
const std::string ImageStatisticsViewPreferencesPage::BACKGROUND_VALUE_NAME("background value");
#endif

UndistortViewPreferencesPage::UndistortViewPreferencesPage()
#if 0
: m_MainControl(0)
, m_AutoUpdate(0)
, m_AssumeBinary(0)
, m_RequireSameSizeImage(0)
, m_BackgroundValue(0)
#endif
{

}

UndistortViewPreferencesPage::UndistortViewPreferencesPage(const UndistortViewPreferencesPage& other)
: berry::Object(), QObject()
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}

UndistortViewPreferencesPage::~UndistortViewPreferencesPage()
{

}

void UndistortViewPreferencesPage::Init(berry::IWorkbench::Pointer )
{

}

void UndistortViewPreferencesPage::CreateQtControl(QWidget* parent)
{
#if 0
  berry::IPreferencesService::Pointer prefService
    = berry::Platform::GetServiceRegistry()
      .GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);

  m_ImageStatisticsPreferencesNode = prefService->GetSystemPreferences()->Node("/uk.ac.ucl.cmic.imagestatistics");

  m_MainControl = new QWidget(parent);

  m_AutoUpdate = new QCheckBox(m_MainControl);
  m_AssumeBinary = new QCheckBox(m_MainControl);
  m_RequireSameSizeImage = new QCheckBox(m_MainControl);
  m_RequireSameSizeImage->setVisible(false); // TODO - make it work by interpolating millimetre positions.
  m_BackgroundValue = new QSpinBox(m_MainControl);
  m_BackgroundValue->setMinimum(-10000);
  m_BackgroundValue->setMaximum(10000);
  m_BackgroundValue->setValue(0);

  QFormLayout *formLayout = new QFormLayout;
  formLayout->addRow( "auto-update:", m_AutoUpdate );
  formLayout->addRow( "assume a binary mask:", m_AssumeBinary );
  //formLayout->addRow( "require same size image and mask:", m_RequireSameSizeImage ); TODO  - make it work by interpolating millimetre positions.
  formLayout->addRow( "background value:", m_BackgroundValue);

  m_MainControl->setLayout(formLayout);
  this->Update();
#endif
}

QWidget* UndistortViewPreferencesPage::GetQtControl() const
{
  return 0;//m_MainControl;
}

bool UndistortViewPreferencesPage::PerformOk()
{
#if 0
  m_ImageStatisticsPreferencesNode->PutBool(AUTO_UPDATE_NAME, m_AutoUpdate->isChecked());
  m_ImageStatisticsPreferencesNode->PutBool(ASSUME_BINARY_NAME, m_AssumeBinary->isChecked());
  m_ImageStatisticsPreferencesNode->PutBool(REQUIRE_SAME_SIZE_IMAGE_NAME, m_RequireSameSizeImage->isChecked());
  m_ImageStatisticsPreferencesNode->PutInt(BACKGROUND_VALUE_NAME, m_BackgroundValue->value());
#endif
  return true;
}

void UndistortViewPreferencesPage::PerformCancel()
{

}

void UndistortViewPreferencesPage::Update()
{
#if 0
  m_AutoUpdate->setChecked(m_ImageStatisticsPreferencesNode->GetBool(AUTO_UPDATE_NAME, false));
  m_AssumeBinary->setChecked(m_ImageStatisticsPreferencesNode->GetBool(ASSUME_BINARY_NAME, true));
  m_RequireSameSizeImage->setChecked(m_ImageStatisticsPreferencesNode->GetBool(REQUIRE_SAME_SIZE_IMAGE_NAME, true));
  m_BackgroundValue->setValue(m_ImageStatisticsPreferencesNode->GetInt(BACKGROUND_VALUE_NAME, 0));
#endif
}
