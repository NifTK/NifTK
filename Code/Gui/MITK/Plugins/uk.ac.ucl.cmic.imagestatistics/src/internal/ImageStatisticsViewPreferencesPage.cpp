/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-28 10:00:55 +0100 (Wed, 28 Sep 2011) $
 Revision          : $Revision: 7379 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "ImageStatisticsViewPreferencesPage.h"

#include <QFormLayout>
#include <QVBoxLayout>
#include <QLabel>
#include <QCheckBox>
#include <QMessageBox>
#include <QSpinBox>

#include <berryIPreferencesService.h>
#include <berryPlatform.h>

const std::string ImageStatisticsViewPreferencesPage::AUTO_UPDATE_NAME("auto update");
const std::string ImageStatisticsViewPreferencesPage::ASSUME_BINARY_NAME("assume binary");
const std::string ImageStatisticsViewPreferencesPage::REQUIRE_SAME_SIZE_IMAGE_NAME("require same size image");
const std::string ImageStatisticsViewPreferencesPage::BACKGROUND_VALUE_NAME("background value");

ImageStatisticsViewPreferencesPage::ImageStatisticsViewPreferencesPage()
: m_MainControl(0)
, m_AutoUpdate(0)
, m_AssumeBinary(0)
, m_RequireSameSizeImage(0)
, m_BackgroundValue(0)
{

}

ImageStatisticsViewPreferencesPage::ImageStatisticsViewPreferencesPage(const ImageStatisticsViewPreferencesPage& other)
: berry::Object(), QObject()
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}

ImageStatisticsViewPreferencesPage::~ImageStatisticsViewPreferencesPage()
{

}

void ImageStatisticsViewPreferencesPage::Init(berry::IWorkbench::Pointer )
{

}

void ImageStatisticsViewPreferencesPage::CreateQtControl(QWidget* parent)
{
  berry::IPreferencesService::Pointer prefService
    = berry::Platform::GetServiceRegistry()
      .GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);

  m_ImageStatisticsPreferencesNode = prefService->GetSystemPreferences()->Node("/uk.ac.ucl.cmic.imagestatistics");

  m_MainControl = new QWidget(parent);

  m_AutoUpdate = new QCheckBox(m_MainControl);
  m_AssumeBinary = new QCheckBox(m_MainControl);
  m_RequireSameSizeImage = new QCheckBox(m_MainControl);
  m_BackgroundValue = new QSpinBox(m_MainControl);
  m_BackgroundValue->setMinimum(-10000);
  m_BackgroundValue->setMaximum(10000);
  m_BackgroundValue->setValue(0);

  QFormLayout *formLayout = new QFormLayout;
  formLayout->addRow( "auto-update:", m_AutoUpdate );
  formLayout->addRow( "assume a binary mask:", m_AssumeBinary );
  //formLayout->addRow( "require same size image and mask:", m_RequireSameSizeImage ); TODO.
  formLayout->addRow( "background value:", m_BackgroundValue);

  m_MainControl->setLayout(formLayout);
  this->Update();
}

QWidget* ImageStatisticsViewPreferencesPage::GetQtControl() const
{
  return m_MainControl;
}

bool ImageStatisticsViewPreferencesPage::PerformOk()
{
  m_ImageStatisticsPreferencesNode->PutBool(AUTO_UPDATE_NAME, m_AutoUpdate->isChecked());
  m_ImageStatisticsPreferencesNode->PutBool(ASSUME_BINARY_NAME, m_AssumeBinary->isChecked());
  m_ImageStatisticsPreferencesNode->PutBool(REQUIRE_SAME_SIZE_IMAGE_NAME, m_RequireSameSizeImage->isChecked());
  m_ImageStatisticsPreferencesNode->PutInt(BACKGROUND_VALUE_NAME, m_BackgroundValue->value());
  return true;
}

void ImageStatisticsViewPreferencesPage::PerformCancel()
{

}

void ImageStatisticsViewPreferencesPage::Update()
{
  m_AutoUpdate->setChecked(m_ImageStatisticsPreferencesNode->GetBool(AUTO_UPDATE_NAME, false));
  m_AssumeBinary->setChecked(m_ImageStatisticsPreferencesNode->GetBool(ASSUME_BINARY_NAME, true));
  m_RequireSameSizeImage->setChecked(m_ImageStatisticsPreferencesNode->GetBool(REQUIRE_SAME_SIZE_IMAGE_NAME, true));
  m_BackgroundValue->setValue(m_ImageStatisticsPreferencesNode->GetInt(BACKGROUND_VALUE_NAME, 0));
}
