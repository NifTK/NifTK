/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date$
 Revision          : $Revision$
 Last modified by  : $Author$

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "QmitkThumbnailViewPreferencePage.h"

#include <QFormLayout>
#include <QVBoxLayout>
#include <QLabel>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QPushButton>
#include <QColorDialog>

#include <berryIPreferencesService.h>
#include <berryPlatform.h>

const std::string QmitkThumbnailViewPreferencePage::THUMBNAIL_BOX_COLOUR("thumbnail view box colour");
const std::string QmitkThumbnailViewPreferencePage::THUMBNAIL_BOX_COLOUR_STYLE_SHEET("thumbnail view box colour style sheet");
const std::string QmitkThumbnailViewPreferencePage::THUMBNAIL_BOX_THICKNESS("thumbnail view box thickness");
const std::string QmitkThumbnailViewPreferencePage::THUMBNAIL_BOX_OPACITY("thumbnail view box opacity");
const std::string QmitkThumbnailViewPreferencePage::THUMBNAIL_BOX_LAYER("thumbnail view box layer");

QmitkThumbnailViewPreferencePage::QmitkThumbnailViewPreferencePage()
: m_MainControl(0)
, m_BoxThickness(0)
, m_BoxOpacity(0)
, m_BoxLayer(0)
, m_Initializing(false)
{

}

QmitkThumbnailViewPreferencePage::QmitkThumbnailViewPreferencePage(const QmitkThumbnailViewPreferencePage& other)
: berry::Object(), QObject()
{
  Q_UNUSED(other)
  throw std::runtime_error("QmitkThumbnailViewPreferencePage copy constructor not implemented");
}

QmitkThumbnailViewPreferencePage::~QmitkThumbnailViewPreferencePage()
{

}

void QmitkThumbnailViewPreferencePage::Init(berry::IWorkbench::Pointer )
{

}

void QmitkThumbnailViewPreferencePage::CreateQtControl(QWidget* parent)
{
  m_Initializing = true;

  berry::IPreferencesService::Pointer prefService
    = berry::Platform::GetServiceRegistry()
      .GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);

  m_ThumbnailPreferencesNode = prefService->GetSystemPreferences()->Node("/uk.ac.ucl.cmic.thumbnail");

  m_MainControl = new QWidget(parent);

  m_BoxThickness = new QSpinBox();
  m_BoxThickness->setMinimum(1);
  m_BoxThickness->setMaximum(50);
  m_BoxThickness->setSingleStep(1);
  m_BoxThickness->setValue(1);

  m_BoxOpacity = new QDoubleSpinBox();
  m_BoxOpacity->setMinimum(0);
  m_BoxOpacity->setMaximum(1);
  m_BoxOpacity->setSingleStep(0.1);
  m_BoxOpacity->setValue(1);

  m_BoxLayer = new QSpinBox();
  m_BoxLayer->setMinimum(0);
  m_BoxLayer->setMaximum(1000);
  m_BoxLayer->setSingleStep(1);
  m_BoxLayer->setValue(99);

  QFormLayout *formLayout = new QFormLayout;
  formLayout->addRow("line width", m_BoxThickness );
  formLayout->addRow("line opacity", m_BoxOpacity );

  QPushButton* boxColorResetButton = new QPushButton;
  boxColorResetButton->setText("reset");

  m_BoxColorPushButton = new QPushButton;
  m_BoxColorPushButton->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Minimum);

  QGridLayout* boxColorLayout = new QGridLayout;
  boxColorLayout->setContentsMargins(4,4,4,4);
  boxColorLayout->addWidget(m_BoxColorPushButton, 0, 0);
  boxColorLayout->addWidget(boxColorResetButton, 0, 1);

  formLayout->addRow("line colour", boxColorLayout);
  formLayout->addRow("rendering layer", m_BoxLayer );

  m_MainControl->setLayout(formLayout);

  QObject::connect( m_BoxColorPushButton, SIGNAL( clicked() ), this, SLOT( OnBoxColourChanged() ) );
  QObject::connect( boxColorResetButton, SIGNAL( clicked() ), this, SLOT( OnResetBoxColour() ) );

  this->Update();

  m_Initializing = false;
}

QWidget* QmitkThumbnailViewPreferencePage::GetQtControl() const
{
  return m_MainControl;
}

bool QmitkThumbnailViewPreferencePage::PerformOk()
{
  m_ThumbnailPreferencesNode->Put(THUMBNAIL_BOX_COLOUR_STYLE_SHEET, m_BoxColorStyleSheet.toStdString());
  m_ThumbnailPreferencesNode->PutByteArray(THUMBNAIL_BOX_COLOUR, m_BoxColor);
  m_ThumbnailPreferencesNode->PutDouble(THUMBNAIL_BOX_OPACITY, m_BoxOpacity->value());
  m_ThumbnailPreferencesNode->PutInt(THUMBNAIL_BOX_THICKNESS, m_BoxThickness->value());
  m_ThumbnailPreferencesNode->PutInt(THUMBNAIL_BOX_LAYER, m_BoxLayer->value());
  return true;
}

void QmitkThumbnailViewPreferencePage::PerformCancel()
{

}

void QmitkThumbnailViewPreferencePage::Update()
{
  m_BoxColorStyleSheet = QString::fromStdString(m_ThumbnailPreferencesNode->Get(THUMBNAIL_BOX_COLOUR_STYLE_SHEET, ""));
  m_BoxColor = m_ThumbnailPreferencesNode->GetByteArray(THUMBNAIL_BOX_COLOUR, "");
  if (m_BoxColorStyleSheet=="")
  {
    m_BoxColorStyleSheet = "background-color: rgb(255,0,0)";
  }
  if (m_BoxColor=="")
  {
    m_BoxColor = "#ff0000";
  }
  m_BoxColorPushButton->setStyleSheet(m_BoxColorStyleSheet);

  m_BoxThickness->setValue(m_ThumbnailPreferencesNode->GetInt(THUMBNAIL_BOX_THICKNESS, 1));
  m_BoxLayer->setValue(m_ThumbnailPreferencesNode->GetInt(THUMBNAIL_BOX_LAYER, 99));
  m_BoxOpacity->setValue(m_ThumbnailPreferencesNode->GetDouble(THUMBNAIL_BOX_OPACITY, 1));
}

void QmitkThumbnailViewPreferencePage::OnBoxColourChanged()
{
  QColor colour = QColorDialog::getColor();
  if (colour.isValid())
  {
    m_BoxColorPushButton->setAutoFillBackground(true);

    QString styleSheet = "background-color: rgb(";
    styleSheet.append(QString::number(colour.red()));
    styleSheet.append(",");
    styleSheet.append(QString::number(colour.green()));
    styleSheet.append(",");
    styleSheet.append(QString::number(colour.blue()));
    styleSheet.append(")");

    m_BoxColorPushButton->setStyleSheet(styleSheet);
    m_BoxColorStyleSheet = styleSheet;

    QStringList boxColor;
    boxColor << colour.name();

    m_BoxColor = boxColor.replaceInStrings(";","\\;").join(";").toStdString();
  }
 }

void QmitkThumbnailViewPreferencePage::OnResetBoxColour()
{
  m_BoxColorStyleSheet = "background-color: rgb(255,0,0)";
  m_BoxColor = "#ff0000";
  m_BoxColorPushButton->setStyleSheet(m_BoxColorStyleSheet);
}
