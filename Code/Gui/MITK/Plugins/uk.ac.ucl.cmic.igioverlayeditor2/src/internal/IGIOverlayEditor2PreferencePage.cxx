/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "IGIOverlayEditor2PreferencePage.h"
#include <IGIOverlayEditor2.h>

#include <QLabel>
#include <QPushButton>
#include <QFormLayout>
#include <QRadioButton>
#include <QColorDialog>
#include <QCheckBox>
#include <ctkPathLineEdit.h>
#include <berryIPreferencesService.h>
#include <berryPlatform.h>

const std::string IGIOverlayEditor2PreferencePage::FIRST_BACKGROUND_STYLE_SHEET("first background color style sheet");
const std::string IGIOverlayEditor2PreferencePage::SECOND_BACKGROUND_STYLE_SHEET("second background color style sheet");
const std::string IGIOverlayEditor2PreferencePage::FIRST_BACKGROUND_COLOUR("first background color");
const std::string IGIOverlayEditor2PreferencePage::SECOND_BACKGROUND_COLOUR("second background color");
const std::string IGIOverlayEditor2PreferencePage::CALIBRATION_FILE_NAME("calibration file name");
const std::string IGIOverlayEditor2PreferencePage::CAMERA_TRACKING_MODE("camera tracking mode");
const std::string IGIOverlayEditor2PreferencePage::CLIP_TO_IMAGE_PLANE("clip to imae plane");

//-----------------------------------------------------------------------------
IGIOverlayEditor2PreferencePage::IGIOverlayEditor2PreferencePage()
: m_MainControl(0)
, m_CameraTrackingMode(NULL)
, m_ImageTrackingMode(NULL)
, m_ColorButton1(NULL)
, m_ColorButton2(NULL)
, m_ClipToImagePlane(NULL)
{
}


//-----------------------------------------------------------------------------
void IGIOverlayEditor2PreferencePage::Init(berry::IWorkbench::Pointer )
{
}


//-----------------------------------------------------------------------------
void IGIOverlayEditor2PreferencePage::CreateQtControl(QWidget* parent)
{
  berry::IPreferencesService::Pointer prefService
    = berry::Platform::GetServiceRegistry()
    .GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);

  m_IGIOverlayEditor2PreferencesNode = prefService->GetSystemPreferences()->Node(IGIOverlayEditor2::EDITOR_ID);

  m_MainControl = new QWidget(parent);

  QFormLayout *formLayout = new QFormLayout;

  m_ImageTrackingMode = new QRadioButton();
  formLayout->addRow("image tracking mode", m_ImageTrackingMode);

  m_ClipToImagePlane = new QCheckBox();
  formLayout->addRow("image tracking clipping planes", m_ClipToImagePlane);
  
  m_CameraTrackingMode = new QRadioButton();
  formLayout->addRow("camera tracking mode", m_CameraTrackingMode);

  m_CalibrationFileName = new ctkPathLineEdit();
  formLayout->addRow("hand-eye calibration transform", m_CalibrationFileName);

  // gradient background
  QLabel* gBName = new QLabel;
  gBName->setText("gradient background");
  formLayout->addRow(gBName);

  // color
  m_ColorButton1 = new QPushButton;
  m_ColorButton1->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Minimum);

  m_ColorButton2 = new QPushButton;
  m_ColorButton2->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Minimum);

  QPushButton* resetButton = new QPushButton;
  resetButton->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Minimum);
  resetButton->setText("reset");

  QLabel* colorLabel1 = new QLabel("first color : ");
  colorLabel1->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::Minimum);

  QLabel* colorLabel2 = new QLabel("second color: ");
  colorLabel2->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::Minimum);

  QHBoxLayout* colorWidgetLayout = new QHBoxLayout;
  colorWidgetLayout->setContentsMargins(4,4,4,4);
  colorWidgetLayout->addWidget(colorLabel1);
  colorWidgetLayout->addWidget(m_ColorButton1);
  colorWidgetLayout->addWidget(colorLabel2);
  colorWidgetLayout->addWidget(m_ColorButton2);
  colorWidgetLayout->addWidget(resetButton);

  QWidget* colorWidget = new QWidget;
  colorWidget->setLayout(colorWidgetLayout);

  // spacer
  QSpacerItem *spacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);
  QVBoxLayout* vBoxLayout = new QVBoxLayout;
  vBoxLayout->addLayout(formLayout);
  vBoxLayout->addWidget(colorWidget);
  vBoxLayout->addSpacerItem(spacer);

  m_MainControl->setLayout(vBoxLayout);

  QObject::connect( m_ColorButton1, SIGNAL( clicked() )
    , this, SLOT( FirstColorChanged() ) );

  QObject::connect( m_ColorButton2, SIGNAL( clicked() )
    , this, SLOT( SecondColorChanged() ) );

  QObject::connect( resetButton, SIGNAL( clicked() )
    , this, SLOT( ResetColors() ) );

  this->Update();
}


//-----------------------------------------------------------------------------
QWidget* IGIOverlayEditor2PreferencePage::GetQtControl() const
{
  return m_MainControl;
}


//-----------------------------------------------------------------------------
bool IGIOverlayEditor2PreferencePage::PerformOk()
{
  m_IGIOverlayEditor2PreferencesNode->Put(IGIOverlayEditor2PreferencePage::FIRST_BACKGROUND_STYLE_SHEET, m_FirstColorStyleSheet.toStdString());
  m_IGIOverlayEditor2PreferencesNode->Put(IGIOverlayEditor2PreferencePage::SECOND_BACKGROUND_STYLE_SHEET, m_SecondColorStyleSheet.toStdString());
  m_IGIOverlayEditor2PreferencesNode->PutByteArray(IGIOverlayEditor2PreferencePage::FIRST_BACKGROUND_COLOUR, m_FirstColor);
  m_IGIOverlayEditor2PreferencesNode->PutByteArray(IGIOverlayEditor2PreferencePage::SECOND_BACKGROUND_COLOUR, m_SecondColor);
  m_IGIOverlayEditor2PreferencesNode->Put(IGIOverlayEditor2PreferencePage::CALIBRATION_FILE_NAME, m_CalibrationFileName->currentPath().toStdString());
  m_IGIOverlayEditor2PreferencesNode->PutBool(IGIOverlayEditor2PreferencePage::CAMERA_TRACKING_MODE, m_CameraTrackingMode->isChecked());
  m_IGIOverlayEditor2PreferencesNode->PutBool(IGIOverlayEditor2PreferencePage::CLIP_TO_IMAGE_PLANE, m_ClipToImagePlane->isChecked());
  return true;
}


//-----------------------------------------------------------------------------
void IGIOverlayEditor2PreferencePage::PerformCancel()
{

}


//-----------------------------------------------------------------------------
void IGIOverlayEditor2PreferencePage::Update()
{
  m_FirstColorStyleSheet = QString::fromStdString(m_IGIOverlayEditor2PreferencesNode->Get(IGIOverlayEditor2PreferencePage::FIRST_BACKGROUND_STYLE_SHEET, ""));
  m_SecondColorStyleSheet = QString::fromStdString(m_IGIOverlayEditor2PreferencesNode->Get(IGIOverlayEditor2PreferencePage::SECOND_BACKGROUND_STYLE_SHEET, ""));
  m_FirstColor = m_IGIOverlayEditor2PreferencesNode->GetByteArray(IGIOverlayEditor2PreferencePage::FIRST_BACKGROUND_COLOUR, "");
  m_SecondColor = m_IGIOverlayEditor2PreferencesNode->GetByteArray(IGIOverlayEditor2PreferencePage::SECOND_BACKGROUND_COLOUR, "");
  if (m_FirstColorStyleSheet=="")
  {
    m_FirstColorStyleSheet = "background-color:rgb(0,0,0)";
  }
  if (m_SecondColorStyleSheet=="")
  {
    m_SecondColorStyleSheet = "background-color:rgb(0,0,0)";
  }
  if (m_FirstColor=="")
  {
    m_FirstColor = "#000000";
  }
  if (m_SecondColor=="")
  {
    m_SecondColor = "#000000";
  }
  m_ColorButton1->setStyleSheet(m_FirstColorStyleSheet);
  m_ColorButton2->setStyleSheet(m_SecondColorStyleSheet);
  m_CalibrationFileName->setCurrentPath(QString::fromStdString(m_IGIOverlayEditor2PreferencesNode->Get(IGIOverlayEditor2PreferencePage::CALIBRATION_FILE_NAME, "")));

  m_ClipToImagePlane->setChecked(m_IGIOverlayEditor2PreferencesNode->GetBool(IGIOverlayEditor2PreferencePage::CLIP_TO_IMAGE_PLANE, true));
  bool isCameraTracking = m_IGIOverlayEditor2PreferencesNode->GetBool(IGIOverlayEditor2PreferencePage::CAMERA_TRACKING_MODE, true);
  m_CameraTrackingMode->setChecked(isCameraTracking);
  m_ImageTrackingMode->setChecked(!isCameraTracking);
}


//-----------------------------------------------------------------------------
void IGIOverlayEditor2PreferencePage::FirstColorChanged()
{
  QColor color = QColorDialog::getColor();
  m_ColorButton1->setAutoFillBackground(true);
  QString styleSheet = "background-color:rgb(";

  styleSheet.append(QString::number(color.red()));
  styleSheet.append(",");
  styleSheet.append(QString::number(color.green()));
  styleSheet.append(",");
  styleSheet.append(QString::number(color.blue()));
  styleSheet.append(")");
  m_ColorButton1->setStyleSheet(styleSheet);

  m_FirstColorStyleSheet = styleSheet;
  QStringList firstColor;
  firstColor << color.name();
  m_FirstColor = firstColor.replaceInStrings(";","\\;").join(";").toStdString();
 }


//-----------------------------------------------------------------------------
void IGIOverlayEditor2PreferencePage::SecondColorChanged()
{
  QColor color = QColorDialog::getColor();
  m_ColorButton2->setAutoFillBackground(true);
  QString styleSheet = "background-color:rgb(";

  styleSheet.append(QString::number(color.red()));
  styleSheet.append(",");
  styleSheet.append(QString::number(color.green()));
  styleSheet.append(",");
  styleSheet.append(QString::number(color.blue()));
  styleSheet.append(")");
  m_ColorButton2->setStyleSheet(styleSheet);

  m_SecondColorStyleSheet = styleSheet;
  QStringList secondColor;
  secondColor << color.name();
  m_SecondColor = secondColor.replaceInStrings(";","\\;").join(";").toStdString();
 }


//-----------------------------------------------------------------------------
void IGIOverlayEditor2PreferencePage::ResetColors()
{
  m_FirstColorStyleSheet = "background-color:rgb(0,0,0)";
  m_SecondColorStyleSheet = "background-color:rgb(0,0,0)";
  m_FirstColor = "#000000";
  m_SecondColor = "#000000";
  m_ColorButton1->setStyleSheet(m_FirstColorStyleSheet);
  m_ColorButton2->setStyleSheet(m_SecondColorStyleSheet);
}

