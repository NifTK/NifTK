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
const QString CameraCalViewPreferencePage::ITERATIVE_NODE_NAME("iterative");
const QString CameraCalViewPreferencePage::MINIMUM_VIEWS_NODE_NAME("minimum number of views");
const QString CameraCalViewPreferencePage::MODEL_NODE_NAME("3D model points");
const QString CameraCalViewPreferencePage::SCALEX_NODE_NAME("scale factor in x to resize image");
const QString CameraCalViewPreferencePage::SCALEY_NODE_NAME("scale factor in y to resize image");
const QString CameraCalViewPreferencePage::PATTERN_NODE_NAME("pattern");
const QString CameraCalViewPreferencePage::TAG_FAMILY_NODE_NAME("tag family");
const QString CameraCalViewPreferencePage::GRIDX_NODE_NAME("grid size in x");
const QString CameraCalViewPreferencePage::GRIDY_NODE_NAME("grid size in y");
const QString CameraCalViewPreferencePage::HANDEYE_NODE_NAME("handeye method");
const QString CameraCalViewPreferencePage::MODEL_TO_TRACKER_NODE_NAME("model to tracker transform");
const QString CameraCalViewPreferencePage::REFERENCE_IMAGE_NODE_NAME("reference image");
const QString CameraCalViewPreferencePage::REFERENCE_POINTS_NODE_NAME("reference points");

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
  ok = connect(m_Ui->m_ReferenceImagePushButton, SIGNAL(pressed()), this, SLOT(OnReferenceImageButtonPressed()));
  assert(ok);
  ok = connect(m_Ui->m_ReferencePointsPushButton, SIGNAL(pressed()), this, SLOT(OnReferencePointsButtonPressed()));
  assert(ok);
  ok = connect(m_Ui->m_IterativeCheckBox, SIGNAL(clicked(bool)), this, SLOT(OnIterativeCheckBoxChecked(bool)));
  assert(ok);

  m_Ui->m_FeaturesComboBox->setCurrentIndex(2);
  m_Ui->m_TagFamilyComboBox->setCurrentIndex(1);
  m_Ui->m_IterativeCheckBox->setChecked(false);
  this->OnIterativeCheckBoxChecked(false);

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
      m_Ui->m_GridSizeLabel->setVisible(true);
      m_Ui->m_GridPointsInXSpinBox->setVisible(true);
      m_Ui->m_ByLabel->setVisible(true);
      m_Ui->m_GridPointsInYSpinBox->setVisible(true);
      m_Ui->m_TagFamilyComboBox->setVisible(false);
      m_Ui->m_TagFamilyLabel->setVisible(false);
    break;

    case niftk::NiftyCalVideoCalibrationManager::APRIL_TAGS:
      m_Ui->m_GridSizeLabel->setVisible(false);
      m_Ui->m_GridPointsInXSpinBox->setVisible(false);
      m_Ui->m_ByLabel->setVisible(false);
      m_Ui->m_GridPointsInYSpinBox->setVisible(false);
      m_Ui->m_TagFamilyComboBox->setVisible(true);
      m_Ui->m_TagFamilyLabel->setVisible(true);
    break;
  }
}


//-----------------------------------------------------------------------------
void CameraCalViewPreferencePage::On3DModelButtonPressed()
{
  QString fileName = QFileDialog::getOpenFileName(m_Control,
      tr("3D Model File"), "", tr("3D Model Files (*.txt)"));

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
void CameraCalViewPreferencePage::OnReferenceImageButtonPressed()
{
  QString fileName = QFileDialog::getOpenFileName(m_Control,
      tr("Reference image"), "", tr("Image Files (*.jpg *.png *.bmp)"));

  if (!fileName.isEmpty())
  {
    m_Ui->m_ReferenceImageLineEdit->setText(fileName);
  }
}


//-----------------------------------------------------------------------------
void CameraCalViewPreferencePage::OnReferencePointsButtonPressed()
{
  QString fileName = QFileDialog::getOpenFileName(m_Control,
      tr("Reference points"), "", tr("2D Model Files (*.txt)"));

  if (!fileName.isEmpty())
  {
    m_Ui->m_ReferencePointsLineEdit->setText(fileName);
  }
}


//-----------------------------------------------------------------------------
void CameraCalViewPreferencePage::OnIterativeCheckBoxChecked(bool checked)
{
  if (checked)
  {
    m_Ui->m_ReferenceImageLabel->setVisible(true);
    m_Ui->m_ReferenceImageLineEdit->setVisible(true);
    m_Ui->m_ReferenceImagePushButton->setVisible(true);
    m_Ui->m_ReferencePointsLabel->setVisible(true);
    m_Ui->m_ReferencePointsLineEdit->setVisible(true);
    m_Ui->m_ReferencePointsPushButton->setVisible(true);
  }
  else
  {
    m_Ui->m_ReferenceImageLabel->setVisible(false);
    m_Ui->m_ReferenceImageLineEdit->setVisible(false);
    m_Ui->m_ReferenceImagePushButton->setVisible(false);
    m_Ui->m_ReferencePointsLabel->setVisible(false);
    m_Ui->m_ReferencePointsLineEdit->setVisible(false);
    m_Ui->m_ReferencePointsPushButton->setVisible(false);
  }
}


//-----------------------------------------------------------------------------
bool CameraCalViewPreferencePage::PerformOk()
{
  m_CameraCalViewPreferencesNode->PutBool(CameraCalViewPreferencePage::ITERATIVE_NODE_NAME, m_Ui->m_IterativeCheckBox->isChecked());
  m_CameraCalViewPreferencesNode->Put(CameraCalViewPreferencePage::MODEL_NODE_NAME, m_Ui->m_3DModelLineEdit->text());
  m_CameraCalViewPreferencesNode->PutInt(CameraCalViewPreferencePage::MINIMUM_VIEWS_NODE_NAME, m_Ui->m_MinimumViewsSpinBox->value());
  m_CameraCalViewPreferencesNode->PutDouble(CameraCalViewPreferencePage::SCALEX_NODE_NAME, m_Ui->m_ScaleImageInXSpinBox->value());
  m_CameraCalViewPreferencesNode->PutDouble(CameraCalViewPreferencePage::SCALEY_NODE_NAME, m_Ui->m_ScaleImageInYSpinBox->value());
  m_CameraCalViewPreferencesNode->PutInt(CameraCalViewPreferencePage::GRIDX_NODE_NAME, m_Ui->m_GridPointsInXSpinBox->value());
  m_CameraCalViewPreferencesNode->PutInt(CameraCalViewPreferencePage::GRIDY_NODE_NAME, m_Ui->m_GridPointsInYSpinBox->value());
  m_CameraCalViewPreferencesNode->Put(CameraCalViewPreferencePage::MODEL_TO_TRACKER_NODE_NAME, m_Ui->m_ModelToTrackerLineEdit->text());
  m_CameraCalViewPreferencesNode->Put(CameraCalViewPreferencePage::REFERENCE_IMAGE_NODE_NAME, m_Ui->m_ReferenceImageLineEdit->text());
  m_CameraCalViewPreferencesNode->Put(CameraCalViewPreferencePage::REFERENCE_POINTS_NODE_NAME, m_Ui->m_ReferencePointsLineEdit->text());
  m_CameraCalViewPreferencesNode->Put(CameraCalViewPreferencePage::TAG_FAMILY_NODE_NAME, m_Ui->m_TagFamilyComboBox->currentText());
  m_CameraCalViewPreferencesNode->PutInt(CameraCalViewPreferencePage::PATTERN_NODE_NAME, m_Ui->m_FeaturesComboBox->currentIndex());
  m_CameraCalViewPreferencesNode->PutInt(CameraCalViewPreferencePage::HANDEYE_NODE_NAME, m_Ui->m_HandEyeComboBox->currentIndex());
  return true;
}


//-----------------------------------------------------------------------------
void CameraCalViewPreferencePage::PerformCancel()
{
}


//-----------------------------------------------------------------------------
void CameraCalViewPreferencePage::Update()
{
  m_Ui->m_IterativeCheckBox->setChecked(m_CameraCalViewPreferencesNode->GetBool(CameraCalViewPreferencePage::ITERATIVE_NODE_NAME, niftk::NiftyCalVideoCalibrationManager::DefaultDoIterative));
  m_Ui->m_3DModelLineEdit->setText(m_CameraCalViewPreferencesNode->Get(CameraCalViewPreferencePage::MODEL_NODE_NAME, ""));
  m_Ui->m_MinimumViewsSpinBox->setValue(m_CameraCalViewPreferencesNode->GetInt(CameraCalViewPreferencePage::MINIMUM_VIEWS_NODE_NAME, niftk::NiftyCalVideoCalibrationManager::DefaultMinimumNumberOfSnapshotsForCalibrating));
  m_Ui->m_ScaleImageInXSpinBox->setValue(m_CameraCalViewPreferencesNode->GetDouble(CameraCalViewPreferencePage::SCALEX_NODE_NAME, niftk::NiftyCalVideoCalibrationManager::DefaultScaleFactorX));
  m_Ui->m_ScaleImageInYSpinBox->setValue(m_CameraCalViewPreferencesNode->GetDouble(CameraCalViewPreferencePage::SCALEY_NODE_NAME, niftk::NiftyCalVideoCalibrationManager::DefaultScaleFactorY));
  m_Ui->m_GridPointsInXSpinBox->setValue(m_CameraCalViewPreferencesNode->GetInt(CameraCalViewPreferencePage::GRIDX_NODE_NAME, niftk::NiftyCalVideoCalibrationManager::DefaultGridSizeX));
  m_Ui->m_GridPointsInYSpinBox->setValue(m_CameraCalViewPreferencesNode->GetInt(CameraCalViewPreferencePage::GRIDY_NODE_NAME, niftk::NiftyCalVideoCalibrationManager::DefaultGridSizeY));
  m_Ui->m_ModelToTrackerLineEdit->setText(m_CameraCalViewPreferencesNode->Get(CameraCalViewPreferencePage::MODEL_TO_TRACKER_NODE_NAME, ""));
  m_Ui->m_ReferenceImageLineEdit->setText(m_CameraCalViewPreferencesNode->Get(CameraCalViewPreferencePage::REFERENCE_IMAGE_NODE_NAME, ""));
  m_Ui->m_ReferencePointsLineEdit->setText(m_CameraCalViewPreferencesNode->Get(CameraCalViewPreferencePage::REFERENCE_POINTS_NODE_NAME, ""));
  m_Ui->m_TagFamilyComboBox->setCurrentIndex(
        m_Ui->m_TagFamilyComboBox->findText(m_CameraCalViewPreferencesNode->Get(CameraCalViewPreferencePage::TAG_FAMILY_NODE_NAME, QString::fromStdString(niftk::NiftyCalVideoCalibrationManager::DefaultTagFamily))));
  m_Ui->m_FeaturesComboBox->setCurrentIndex(m_CameraCalViewPreferencesNode->GetInt(CameraCalViewPreferencePage::PATTERN_NODE_NAME, static_cast<int>(niftk::NiftyCalVideoCalibrationManager::DefaultCalibrationPattern)));
  m_Ui->m_HandEyeComboBox->setCurrentIndex(m_CameraCalViewPreferencesNode->GetInt(CameraCalViewPreferencePage::HANDEYE_NODE_NAME, static_cast<int>(niftk::NiftyCalVideoCalibrationManager::DefaultHandEyeMethod)));
}

} // end namespace
