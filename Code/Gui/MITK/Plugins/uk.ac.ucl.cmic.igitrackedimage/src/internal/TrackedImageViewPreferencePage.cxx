/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "TrackedImageViewPreferencePage.h"

#include <QFormLayout>
#include <QVBoxLayout>
#include <QLabel>
#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QMessageBox>
#include <QPushButton>
#include <ctkPathLineEdit.h>
#include <QCheckBox>

#include <berryIPreferencesService.h>
#include <berryPlatform.h>

const std::string TrackedImageViewPreferencePage::PREFERENCES_NODE_NAME("/uk.ac.ucl.cmic.igitrackedimage");
const std::string TrackedImageViewPreferencePage::CALIBRATION_FILE_NAME("calibration file name");
const std::string TrackedImageViewPreferencePage::SCALE_FILE_NAME("scale file name");
const std::string TrackedImageViewPreferencePage::EMTOWORLDCALIBRATION_FILE_NAME("Em to optical calibration file name");
const std::string TrackedImageViewPreferencePage::FLIP_X_SCALING("flip x scaling");
const std::string TrackedImageViewPreferencePage::FLIP_Y_SCALING("flip y scaling");
const std::string TrackedImageViewPreferencePage::CLONE_IMAGE("Clone image");


//-----------------------------------------------------------------------------
TrackedImageViewPreferencePage::TrackedImageViewPreferencePage()
: m_MainControl(0)
, m_CalibrationFileName(0)
, m_ScaleFileName(0)
, m_EmToWorldCalibrationFileName(0)
, m_Initializing(false)
, m_CloneImage(0)
, m_TrackedImageViewPreferencesNode(0)
{
}


//-----------------------------------------------------------------------------
TrackedImageViewPreferencePage::TrackedImageViewPreferencePage(const TrackedImageViewPreferencePage& other)
: berry::Object(), QObject()
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
TrackedImageViewPreferencePage::~TrackedImageViewPreferencePage()
{
}


//-----------------------------------------------------------------------------
void TrackedImageViewPreferencePage::Init(berry::IWorkbench::Pointer )
{

}


//-----------------------------------------------------------------------------
void TrackedImageViewPreferencePage::CreateQtControl(QWidget* parent)
{
  m_Initializing = true;

  berry::IPreferencesService::Pointer prefService
    = berry::Platform::GetServiceRegistry()
      .GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);

  m_TrackedImageViewPreferencesNode = prefService->GetSystemPreferences()->Node(PREFERENCES_NODE_NAME);

  m_MainControl = new QWidget(parent);
  QFormLayout *formLayout = new QFormLayout;

  m_CalibrationFileName = new ctkPathLineEdit();
  formLayout->addRow("calibration matrix file name", m_CalibrationFileName);

  m_ScaleFileName = new ctkPathLineEdit();
  formLayout->addRow("scale matrix file name", m_ScaleFileName);

  m_EmToWorldCalibrationFileName = new ctkPathLineEdit();
  formLayout->addRow("EM to optical calibration matrix file name", m_EmToWorldCalibrationFileName);

  m_FlipXScaling = new QCheckBox();
  m_FlipXScaling->setChecked(false);
  formLayout->addRow("flip x scale factor", m_FlipXScaling);

  m_FlipYScaling = new QCheckBox();
  m_FlipYScaling->setChecked(false);
  formLayout->addRow("flip y scale factor", m_FlipYScaling);

  m_CloneImage = new QCheckBox();
  m_CloneImage->setChecked(false);
  formLayout->addRow("show clone image button", m_CloneImage);

  m_MainControl->setLayout(formLayout);
  this->Update();

  m_Initializing = false;
}


//-----------------------------------------------------------------------------
QWidget* TrackedImageViewPreferencePage::GetQtControl() const
{
  return m_MainControl;
}


//-----------------------------------------------------------------------------
bool TrackedImageViewPreferencePage::PerformOk()
{
  m_TrackedImageViewPreferencesNode->Put(CALIBRATION_FILE_NAME, m_CalibrationFileName->currentPath().toStdString());
  m_TrackedImageViewPreferencesNode->Put(SCALE_FILE_NAME, m_ScaleFileName->currentPath().toStdString());
  m_TrackedImageViewPreferencesNode->Put(EMTOWORLDCALIBRATION_FILE_NAME, m_EmToWorldCalibrationFileName->currentPath().toStdString());
  m_TrackedImageViewPreferencesNode->PutBool(FLIP_X_SCALING, m_FlipXScaling->isChecked());
  m_TrackedImageViewPreferencesNode->PutBool(FLIP_Y_SCALING, m_FlipYScaling->isChecked());
  m_TrackedImageViewPreferencesNode->PutBool(CLONE_IMAGE, m_CloneImage->isChecked());
  return true;
}


//-----------------------------------------------------------------------------
void TrackedImageViewPreferencePage::PerformCancel()
{

}


//-----------------------------------------------------------------------------
void TrackedImageViewPreferencePage::Update()
{
  m_CalibrationFileName->setCurrentPath(QString(m_TrackedImageViewPreferencesNode->Get(CALIBRATION_FILE_NAME, "").c_str()));
  m_ScaleFileName->setCurrentPath(QString(m_TrackedImageViewPreferencesNode->Get(SCALE_FILE_NAME, "").c_str()));
  m_EmToWorldCalibrationFileName->setCurrentPath(QString(m_TrackedImageViewPreferencesNode->Get(EMTOWORLDCALIBRATION_FILE_NAME, "").c_str()));
  m_FlipXScaling->setChecked(m_TrackedImageViewPreferencesNode->GetBool(FLIP_X_SCALING, false));
  m_FlipYScaling->setChecked(m_TrackedImageViewPreferencesNode->GetBool(FLIP_Y_SCALING, false));
  m_CloneImage->setChecked(m_TrackedImageViewPreferencesNode->GetBool(CLONE_IMAGE, false));
}
