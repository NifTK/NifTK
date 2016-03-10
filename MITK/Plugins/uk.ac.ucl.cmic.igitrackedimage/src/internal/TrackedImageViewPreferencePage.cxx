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

const QString TrackedImageViewPreferencePage::PREFERENCES_NODE_NAME("/uk.ac.ucl.cmic.igitrackedimage");
const QString TrackedImageViewPreferencePage::CALIBRATION_FILE_NAME("calibration file name");
const QString TrackedImageViewPreferencePage::SCALE_FILE_NAME("scale file name");
const QString TrackedImageViewPreferencePage::EMTOWORLDCALIBRATION_FILE_NAME("Em to optical calibration file name");
const QString TrackedImageViewPreferencePage::FLIP_X_SCALING("flip x scaling");
const QString TrackedImageViewPreferencePage::FLIP_Y_SCALING("flip y scaling");
const QString TrackedImageViewPreferencePage::CLONE_IMAGE("Clone image");
const QString TrackedImageViewPreferencePage::SHOW_2D_WINDOW("show 2D window");

//-----------------------------------------------------------------------------
TrackedImageViewPreferencePage::TrackedImageViewPreferencePage()
: m_MainControl(0)
, m_CalibrationFileName(0)
, m_ScaleFileName(0)
, m_EmToWorldCalibrationFileName(0)
, m_Initializing(false)
, m_CloneImage(0)
, m_Show2DWindow(0)
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

  berry::IPreferencesService* prefService = berry::Platform::GetPreferencesService();

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

  m_Show2DWindow = new QCheckBox();
  m_Show2DWindow->setChecked(false);
  formLayout->addRow("show 2D window", m_Show2DWindow);

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
  m_TrackedImageViewPreferencesNode->Put(CALIBRATION_FILE_NAME, m_CalibrationFileName->currentPath());
  m_TrackedImageViewPreferencesNode->Put(SCALE_FILE_NAME, m_ScaleFileName->currentPath());
  m_TrackedImageViewPreferencesNode->Put(EMTOWORLDCALIBRATION_FILE_NAME, m_EmToWorldCalibrationFileName->currentPath());
  m_TrackedImageViewPreferencesNode->PutBool(FLIP_X_SCALING, m_FlipXScaling->isChecked());
  m_TrackedImageViewPreferencesNode->PutBool(FLIP_Y_SCALING, m_FlipYScaling->isChecked());
  m_TrackedImageViewPreferencesNode->PutBool(CLONE_IMAGE, m_CloneImage->isChecked());
  m_TrackedImageViewPreferencesNode->PutBool(SHOW_2D_WINDOW, m_Show2DWindow->isChecked());
  return true;
}


//-----------------------------------------------------------------------------
void TrackedImageViewPreferencePage::PerformCancel()
{

}


//-----------------------------------------------------------------------------
void TrackedImageViewPreferencePage::Update()
{
  m_CalibrationFileName->setCurrentPath(m_TrackedImageViewPreferencesNode->Get(CALIBRATION_FILE_NAME, ""));
  m_ScaleFileName->setCurrentPath(m_TrackedImageViewPreferencesNode->Get(SCALE_FILE_NAME, ""));
  m_EmToWorldCalibrationFileName->setCurrentPath(m_TrackedImageViewPreferencesNode->Get(EMTOWORLDCALIBRATION_FILE_NAME, ""));
  m_FlipXScaling->setChecked(m_TrackedImageViewPreferencesNode->GetBool(FLIP_X_SCALING, false));
  m_FlipYScaling->setChecked(m_TrackedImageViewPreferencesNode->GetBool(FLIP_Y_SCALING, false));
  m_CloneImage->setChecked(m_TrackedImageViewPreferencesNode->GetBool(CLONE_IMAGE, false));
  m_Show2DWindow->setChecked(m_TrackedImageViewPreferencesNode->GetBool(SHOW_2D_WINDOW, false));
}
