/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkImageLookupTablesPreferencePage.h"

#include <QFormLayout>
#include <QVBoxLayout>
#include <QLabel>
#include <QSpinBox>

#include <berryIPreferencesService.h>
#include <berryPlatform.h>


namespace niftk
{

const QString ImageLookupTablesPreferencePage::PRECISION_NAME("precision");

//-----------------------------------------------------------------------------
ImageLookupTablesPreferencePage::ImageLookupTablesPreferencePage()
: m_MainControl(0)
, m_Precision(0)
, m_Initializing(false)
{

}


//-----------------------------------------------------------------------------
ImageLookupTablesPreferencePage::ImageLookupTablesPreferencePage(const ImageLookupTablesPreferencePage& other)
: berry::Object(), QObject()
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
ImageLookupTablesPreferencePage::~ImageLookupTablesPreferencePage()
{

}


//-----------------------------------------------------------------------------
void ImageLookupTablesPreferencePage::Init(berry::IWorkbench::Pointer )
{

}


//-----------------------------------------------------------------------------
void ImageLookupTablesPreferencePage::CreateQtControl(QWidget* parent)
{
  m_Initializing = true;
  berry::IPreferencesService* prefService = berry::Platform::GetPreferencesService();

  m_ImageLookupTablesPreferencesNode = prefService->GetSystemPreferences()->Node("/uk.ac.ucl.cmic.imagelookuptables");

  m_MainControl = new QWidget(parent);

  QFormLayout *formLayout = new QFormLayout;

  QLabel* precisionLabel = new QLabel("Precision:");
  m_Precision = new QSpinBox;
  QString precisionToolTip =
      "Precision of the floating point numbers.";
  precisionLabel->setToolTip(precisionToolTip);
  m_Precision->setToolTip(precisionToolTip);
  formLayout->addRow(precisionLabel, m_Precision);

  m_MainControl->setLayout(formLayout);
  this->Update();

  m_Initializing = false;
}


//-----------------------------------------------------------------------------
QWidget* ImageLookupTablesPreferencePage::GetQtControl() const
{
  return m_MainControl;
}


//-----------------------------------------------------------------------------
bool ImageLookupTablesPreferencePage::PerformOk()
{
  m_ImageLookupTablesPreferencesNode->PutInt(PRECISION_NAME, m_Precision->text().toInt());
  return true;
}


//-----------------------------------------------------------------------------
void ImageLookupTablesPreferencePage::PerformCancel()
{
}


//-----------------------------------------------------------------------------
void ImageLookupTablesPreferencePage::Update()
{
  m_Precision->setValue(m_ImageLookupTablesPreferencesNode->GetInt(PRECISION_NAME, 2));
}

}
