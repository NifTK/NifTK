/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "CameraCalViewPreferencePage.h"
#include "ui_CameraCalViewPreferencePage.h"

#include <berryIPreferencesService.h>
#include <berryPlatform.h>

#include <niftkNiftyCalVideoCalibrationManager.h>

#include <QMessageBox>
#include <QFileDialog>

namespace niftk
{

const QString CameraCalViewPreferencePage::PREFERENCES_NODE_NAME("/uk.ac.ucl.cmic.igicameracal");

//-----------------------------------------------------------------------------
CameraCalViewPreferencePage::CameraCalViewPreferencePage()
: m_Control(nullptr)
, m_Ui(new Ui::CameraCalViewPreferencePage)
, m_Initializing(false)
, m_CameraCalViewPreferencesNode(NULL)
{
}


//-----------------------------------------------------------------------------
CameraCalViewPreferencePage::CameraCalViewPreferencePage(const CameraCalViewPreferencePage& other)
: berry::Object(), QObject()
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
CameraCalViewPreferencePage::~CameraCalViewPreferencePage()
{
}


//-----------------------------------------------------------------------------
void CameraCalViewPreferencePage::Init(berry::IWorkbench::Pointer )
{
}


//-----------------------------------------------------------------------------
void CameraCalViewPreferencePage::CreateQtControl(QWidget* parent)
{
  m_Initializing = true;

  m_Control = new QWidget(parent);
  m_Ui->setupUi(m_Control);

  berry::IPreferencesService* prefService = berry::Platform::GetPreferencesService();
  m_CameraCalViewPreferencesNode = prefService->GetSystemPreferences()->Node(PREFERENCES_NODE_NAME);

  m_Ui->m_FeaturesComboBox->addItem("Chessboard", QVariant(niftk::NiftyCalVideoCalibrationManager::CHESSBOARD));
  m_Ui->m_FeaturesComboBox->addItem("Circles", QVariant(niftk::NiftyCalVideoCalibrationManager::CIRCLE_GRID));
  m_Ui->m_FeaturesComboBox->addItem("April Tags", QVariant(niftk::NiftyCalVideoCalibrationManager::APRIL_TAGS));

  m_Ui->m_TagFamilyComboBox->addItem("16h5");
  m_Ui->m_TagFamilyComboBox->addItem("25h7");
  m_Ui->m_TagFamilyComboBox->addItem("25h9");
  m_Ui->m_TagFamilyComboBox->addItem("36h9");
  m_Ui->m_TagFamilyComboBox->addItem("36h11");

  m_Ui->m_HandEyeComboBox->addItem("Tsai 1989", QVariant(niftk::NiftyCalVideoCalibrationManager::TSAI));
  m_Ui->m_HandEyeComboBox->addItem("Direct", QVariant(niftk::NiftyCalVideoCalibrationManager::DIRECT));
  m_Ui->m_HandEyeComboBox->addItem("Malti 2013", QVariant(niftk::NiftyCalVideoCalibrationManager::MALTI));


  bool ok = false;
  ok = connect(m_Ui->m_FeaturesComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(OnFeaturesComboSelected()));
  assert(ok);
  ok = connect(m_Ui->m_3DModelToolButton, SIGNAL(pressed()), this, SLOT(On3DModelButtonPressed()));
  assert(ok);
  ok = connect(m_Ui->m_ModelToTrackerToolButton, SIGNAL(pressed()), this, SLOT(OnModelToTrackerButtonPressed()));
  assert(ok);
  ok = connect(m_Ui->m_OutputDirectoryToolButton, SIGNAL(pressed()), this, SLOT(OnOutputDirectoryButtonPressed()));
  assert(ok);

  m_Ui->m_FeaturesComboBox->setCurrentIndex(2);
  m_Ui->m_TagFamilyComboBox->setCurrentIndex(1);

  this->Update();

  m_Initializing = false;
}


//-----------------------------------------------------------------------------
QWidget* CameraCalViewPreferencePage::GetQtControl() const
{
  return m_Control;
}


//-----------------------------------------------------------------------------
void CameraCalViewPreferencePage::OnFeaturesComboSelected()
{
  switch(m_Ui->m_FeaturesComboBox->currentIndex())
  {
    case niftk::NiftyCalVideoCalibrationManager::CHESSBOARD:
    case niftk::NiftyCalVideoCalibrationManager::CIRCLE_GRID:
      m_Ui->m_GridSizeLabel->setEnabled(true);
      m_Ui->m_GridSizeLabel->setVisible(true);
      m_Ui->m_GridPointsInXSpinBox->setEnabled(true);
      m_Ui->m_GridPointsInXSpinBox->setVisible(true);
      m_Ui->m_ByLabel->setEnabled(true);
      m_Ui->m_ByLabel->setVisible(true);
      m_Ui->m_GridPointsInYSpinBox->setEnabled(true);
      m_Ui->m_GridPointsInYSpinBox->setVisible(true);
      m_Ui->m_TagFamilyComboBox->setEnabled(false);
      m_Ui->m_TagFamilyComboBox->setVisible(false);
      m_Ui->m_TagFamilyLabel->setEnabled(false);
      m_Ui->m_TagFamilyLabel->setVisible(false);
    break;

    case niftk::NiftyCalVideoCalibrationManager::APRIL_TAGS:
      m_Ui->m_GridSizeLabel->setEnabled(false);
      m_Ui->m_GridSizeLabel->setVisible(false);
      m_Ui->m_GridPointsInXSpinBox->setEnabled(false);
      m_Ui->m_GridPointsInXSpinBox->setVisible(false);
      m_Ui->m_ByLabel->setEnabled(false);
      m_Ui->m_ByLabel->setVisible(false);
      m_Ui->m_GridPointsInYSpinBox->setEnabled(false);
      m_Ui->m_GridPointsInYSpinBox->setVisible(false);
      m_Ui->m_TagFamilyComboBox->setEnabled(true);
      m_Ui->m_TagFamilyComboBox->setVisible(true);
      m_Ui->m_TagFamilyLabel->setEnabled(true);
      m_Ui->m_TagFamilyLabel->setVisible(true);
    break;
  }
}


//-----------------------------------------------------------------------------
void CameraCalViewPreferencePage::On3DModelButtonPressed()
{
  QString fileName = QFileDialog::getOpenFileName(m_Control,
      tr("3D Model File"), "", tr("Model Files (*.txt)"));

  if (!fileName.isEmpty())
  {
    m_Ui->m_3DModelLineEdit->setText(fileName);
  }
}


//-----------------------------------------------------------------------------
void CameraCalViewPreferencePage::OnModelToTrackerButtonPressed()
{
  QString fileName = QFileDialog::getOpenFileName(m_Control,
      tr("Model to tracker transform"), "", tr("Transform (*.4x4 *.txt)"));

  if (!fileName.isEmpty())
  {
    m_Ui->m_ModelToTrackerLineEdit->setText(fileName);
  }
}


//-----------------------------------------------------------------------------
void CameraCalViewPreferencePage::OnOutputDirectoryButtonPressed()
{
  QString dir = QFileDialog::getExistingDirectory(nullptr, tr("Output Dir"),
                                                  "",
                                                  QFileDialog::ShowDirsOnly
                                                  | QFileDialog::DontResolveSymlinks);
  if (!dir.isEmpty())
  {
    m_Ui->m_OutputDirectoryLineEdit->setText(dir);
  }
}


//-----------------------------------------------------------------------------
bool CameraCalViewPreferencePage::PerformOk()
{
  return true;
}


//-----------------------------------------------------------------------------
void CameraCalViewPreferencePage::PerformCancel()
{
}


//-----------------------------------------------------------------------------
void CameraCalViewPreferencePage::Update()
{
}

} // end namespace
