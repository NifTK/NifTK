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

#include <berryIPreferencesService.h>
#include <berryPlatform.h>

const std::string TrackedImageViewPreferencePage::PREFERENCES_NODE_NAME("/uk.ac.ucl.cmic.igitrackedimage");
const std::string TrackedImageViewPreferencePage::CALIBRATION_FILE_NAME("calibration file name");
const std::string TrackedImageViewPreferencePage::X_SCALING("scaling in x (vertical) direction");
const std::string TrackedImageViewPreferencePage::Y_SCALING("scaling in y (vertical) direction");

//-----------------------------------------------------------------------------
TrackedImageViewPreferencePage::TrackedImageViewPreferencePage()
: m_MainControl(0)
, m_CalibrationFileName(0)
, m_Initializing(false)
, m_XScaling(0)
, m_YScaling(0)
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

  m_XScaling = new QDoubleSpinBox();
  m_XScaling->setMinimum(0.00001);
  m_XScaling->setMaximum(10000);
  m_XScaling->setDecimals(4);
  m_XScaling->setSingleStep(0.001);
  formLayout->addRow("scaling in x (horizontal) direction (mm/pix)", m_XScaling);

  m_YScaling = new QDoubleSpinBox();
  m_YScaling->setMinimum(0.00001);
  m_YScaling->setMaximum(10000);
  m_YScaling->setDecimals(4);
  m_YScaling->setSingleStep(0.001);
  formLayout->addRow("scaling in y (vertical) direction (mm/pix)", m_YScaling);

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
  m_TrackedImageViewPreferencesNode->PutDouble(X_SCALING, m_XScaling->value());
  m_TrackedImageViewPreferencesNode->PutDouble(Y_SCALING, m_YScaling->value());
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
  m_XScaling->setValue(m_TrackedImageViewPreferencesNode->GetDouble(X_SCALING, 1));
  m_YScaling->setValue(m_TrackedImageViewPreferencesNode->GetDouble(Y_SCALING, 1));
}
