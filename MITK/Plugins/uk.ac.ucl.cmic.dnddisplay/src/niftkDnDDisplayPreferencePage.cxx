/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkDnDDisplayPreferencePage.h"
#include "niftkMultiViewerEditor.h"

#include <QLabel>
#include <QPushButton>
#include <QFormLayout>
#include <QColorDialog>
#include <QComboBox>
#include <QSpinBox>
#include <QCheckBox>
#include <berryIPreferencesService.h>
#include <berryPlatform.h>

#include "ui_niftkDnDDisplayPreferencePage.h"


namespace niftk
{

const QString DnDDisplayPreferencePage::DNDDISPLAY_DEFAULT_INTERPOLATION_TYPE("default image interpolation");
const QString DnDDisplayPreferencePage::DNDDISPLAY_BACKGROUND_COLOUR("DnD display background colour");
const QString DnDDisplayPreferencePage::DNDDISPLAY_BACKGROUND_COLOUR_STYLESHEET("DnD display background colour stylesheet");

const QString DnDDisplayPreferencePage::DNDDISPLAY_SLICE_SELECT_TRACKING("DnD display slice select tracking");
const QString DnDDisplayPreferencePage::DNDDISPLAY_TIME_SELECT_TRACKING("DnD display time select tracking");
const QString DnDDisplayPreferencePage::DNDDISPLAY_MAGNIFICATION_SELECT_TRACKING("DnD display magnification select tracking");

const QString DnDDisplayPreferencePage::DNDDISPLAY_SHOW_2D_CURSORS("DnD display show 2D cursors");
const QString DnDDisplayPreferencePage::DNDDISPLAY_SHOW_DIRECTION_ANNOTATIONS("DnD display show direction annotations");
const QString DnDDisplayPreferencePage::DNDDISPLAY_SHOW_POSITION_ANNOTATION("DnD display show position annotation");
const QString DnDDisplayPreferencePage::DNDDISPLAY_SHOW_INTENSITY_ANNOTATION("DnD display show intensity annotation");
const QString DnDDisplayPreferencePage::DNDDISPLAY_SHOW_PROPERTY_ANNOTATION("DnD display show property annotation");
const QString DnDDisplayPreferencePage::DNDDISPLAY_PROPERTIES_FOR_ANNOTATION("DnD display properties to show as annotation");

const QString DnDDisplayPreferencePage::DNDDISPLAY_DEFAULT_WINDOW_LAYOUT("DnD display default window layout");
const QString DnDDisplayPreferencePage::DNDDISPLAY_REMEMBER_VIEWER_SETTINGS_PER_WINDOW_LAYOUT("DnD display remember view settings of each window layout");

const QString DnDDisplayPreferencePage::DNDDISPLAY_DEFAULT_VIEWER_ROW_NUMBER("DnD display default number of view rows");
const QString DnDDisplayPreferencePage::DNDDISPLAY_DEFAULT_VIEWER_COLUMN_NUMBER("DnD display default number of view columns");

const QString DnDDisplayPreferencePage::DNDDISPLAY_DEFAULT_DROP_TYPE("DnD display default drop type");

const QString DnDDisplayPreferencePage::DNDDISPLAY_SHOW_MAGNIFICATION_SLIDER("DnD display show magnification slider");
const QString DnDDisplayPreferencePage::DNDDISPLAY_SHOW_SHOWING_OPTIONS("DnD display show showing options");
const QString DnDDisplayPreferencePage::DNDDISPLAY_SHOW_WINDOW_LAYOUT_CONTROLS("DnD display show window layout controls");
const QString DnDDisplayPreferencePage::DNDDISPLAY_SHOW_VIEWER_NUMBER_CONTROLS("DnD display show view number controls");
const QString DnDDisplayPreferencePage::DNDDISPLAY_SHOW_DROP_TYPE_CONTROLS("DnD display show drop type widgets");

//-----------------------------------------------------------------------------
DnDDisplayPreferencePage::DnDDisplayPreferencePage()
: m_MainWidget(nullptr),
  ui(nullptr)
{
}


//-----------------------------------------------------------------------------
DnDDisplayPreferencePage::DnDDisplayPreferencePage(const DnDDisplayPreferencePage& other)
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
void DnDDisplayPreferencePage::Init(berry::IWorkbench::Pointer )
{
}


//-----------------------------------------------------------------------------
void DnDDisplayPreferencePage::CreateQtControl(QWidget* parent)
{
  berry::IPreferencesService* prefService = berry::Platform::GetPreferencesService();

  m_DnDDisplayPreferencesNode = prefService->GetSystemPreferences()->Node(MultiViewerEditor::EDITOR_ID);

  m_MainWidget = new QWidget(parent);

  ui = new Ui::niftkDnDDisplayPreferencePage();
  ui->setupUi(m_MainWidget);

  this->connect(ui->m_BackgroundColourColourButton, SIGNAL( clicked() ), SLOT( OnBackgroundColourChanged() ));
  this->connect(ui->m_BackgroundColourMIDASDefaultButton, SIGNAL( clicked() ), SLOT( OnResetMIDASBackgroundColour() ));
  this->connect(ui->m_BackgroundColourResetButton, SIGNAL( clicked() ), SLOT( OnResetBackgroundColour() ));

  this->Update();
}


//-----------------------------------------------------------------------------
QWidget* DnDDisplayPreferencePage::GetQtControl() const
{
  return m_MainWidget;
}


//-----------------------------------------------------------------------------
bool DnDDisplayPreferencePage::PerformOk()
{
  QString propertiesForAnnotation = ui->m_PropertiesForAnnotationLineEdit->text();
  QStringList properties;
  for (const QString& propertyForAnnotation: propertiesForAnnotation.split(","))
  {
    QString property = propertyForAnnotation.trimmed();
    if (!property.isEmpty())
    {
      properties.push_back(property);
    }
  }
  propertiesForAnnotation = properties.join(", ");

  m_DnDDisplayPreferencesNode->Put(DNDDISPLAY_BACKGROUND_COLOUR_STYLESHEET, m_BackgroundColorStyleSheet);
  m_DnDDisplayPreferencesNode->Put(DNDDISPLAY_BACKGROUND_COLOUR, m_BackgroundColor);
  m_DnDDisplayPreferencesNode->PutInt(DNDDISPLAY_DEFAULT_VIEWER_ROW_NUMBER, ui->m_ViewerRowsSpinBox->value());
  m_DnDDisplayPreferencesNode->PutInt(DNDDISPLAY_DEFAULT_VIEWER_COLUMN_NUMBER, ui->m_ViewerColumnsSpinBox->value());
  m_DnDDisplayPreferencesNode->PutInt(DNDDISPLAY_DEFAULT_WINDOW_LAYOUT, ui->m_WindowLayoutComboBox->currentIndex());
  m_DnDDisplayPreferencesNode->PutInt(DNDDISPLAY_DEFAULT_INTERPOLATION_TYPE, ui->m_ImageInterpolationComboBox->currentIndex());
  m_DnDDisplayPreferencesNode->PutInt(DNDDISPLAY_DEFAULT_DROP_TYPE, ui->m_DefaultDropTypeComboBox->currentIndex());
  m_DnDDisplayPreferencesNode->PutBool(DNDDISPLAY_SHOW_DROP_TYPE_CONTROLS, ui->m_ShowDropTypeControlsCheckBox->isChecked());
  m_DnDDisplayPreferencesNode->PutBool(DNDDISPLAY_SHOW_SHOWING_OPTIONS, ui->m_ShowShowOptionsCheckBox->isChecked());
  m_DnDDisplayPreferencesNode->PutBool(DNDDISPLAY_SHOW_WINDOW_LAYOUT_CONTROLS, ui->m_ShowWindowLayoutControlsCheckBox->isChecked());
  m_DnDDisplayPreferencesNode->PutBool(DNDDISPLAY_SHOW_VIEWER_NUMBER_CONTROLS, ui->m_ShowViewerNumberControlsCheckBox->isChecked());
  m_DnDDisplayPreferencesNode->PutBool(DNDDISPLAY_SHOW_2D_CURSORS, ui->m_Show2DCursorsCheckBox->isChecked());
  m_DnDDisplayPreferencesNode->PutBool(DNDDISPLAY_SHOW_DIRECTION_ANNOTATIONS, ui->m_ShowDirectionAnnotationsCheckBox->isChecked());
  m_DnDDisplayPreferencesNode->PutBool(DNDDISPLAY_SHOW_POSITION_ANNOTATION, ui->m_ShowPositionAnnotationCheckBox->isChecked());
  m_DnDDisplayPreferencesNode->PutBool(DNDDISPLAY_SHOW_INTENSITY_ANNOTATION, ui->m_ShowIntensityAnnotationCheckBox->isChecked());
  m_DnDDisplayPreferencesNode->PutBool(DNDDISPLAY_SHOW_PROPERTY_ANNOTATION, ui->m_ShowPropertyAnnotationCheckBox->isChecked());
  m_DnDDisplayPreferencesNode->Put(DNDDISPLAY_PROPERTIES_FOR_ANNOTATION, propertiesForAnnotation);
  m_DnDDisplayPreferencesNode->PutBool(DNDDISPLAY_SHOW_MAGNIFICATION_SLIDER, ui->m_ShowMagnificationSliderCheckBox->isChecked());
  m_DnDDisplayPreferencesNode->PutBool(DNDDISPLAY_REMEMBER_VIEWER_SETTINGS_PER_WINDOW_LAYOUT, ui->m_RememberViewerSettingsPerWindowLayoutCheckBox->isChecked());
  m_DnDDisplayPreferencesNode->PutBool(DNDDISPLAY_SLICE_SELECT_TRACKING, ui->m_SliceTrackingCheckBox->isChecked());
  m_DnDDisplayPreferencesNode->PutBool(DNDDISPLAY_TIME_SELECT_TRACKING, ui->m_TimeTrackingCheckBox->isChecked());
  m_DnDDisplayPreferencesNode->PutBool(DNDDISPLAY_MAGNIFICATION_SELECT_TRACKING, ui->m_MagnificationTrackingCheckBox->isChecked());
  return true;
}


//-----------------------------------------------------------------------------
void DnDDisplayPreferencePage::PerformCancel()
{
}


//-----------------------------------------------------------------------------
void DnDDisplayPreferencePage::Update()
{
  m_BackgroundColorStyleSheet = m_DnDDisplayPreferencesNode->Get(DNDDISPLAY_BACKGROUND_COLOUR_STYLESHEET, "");
  m_BackgroundColor = m_DnDDisplayPreferencesNode->Get(DNDDISPLAY_BACKGROUND_COLOUR, "");
  if (m_BackgroundColorStyleSheet=="")
  {
    m_BackgroundColorStyleSheet = "background-color: rgb(0, 0, 0)";
  }
  if (m_BackgroundColor=="")
  {
    m_BackgroundColor = "#000000";
  }
  ui->m_BackgroundColourColourButton->setStyleSheet(m_BackgroundColorStyleSheet);

  ui->m_ViewerRowsSpinBox->setValue(m_DnDDisplayPreferencesNode->GetInt(DNDDISPLAY_DEFAULT_VIEWER_ROW_NUMBER, 1));
  ui->m_ViewerColumnsSpinBox->setValue(m_DnDDisplayPreferencesNode->GetInt(DNDDISPLAY_DEFAULT_VIEWER_COLUMN_NUMBER, 1));
  ui->m_WindowLayoutComboBox->setCurrentIndex(m_DnDDisplayPreferencesNode->GetInt(DNDDISPLAY_DEFAULT_WINDOW_LAYOUT, 2)); // default coronal
  ui->m_ImageInterpolationComboBox->setCurrentIndex(m_DnDDisplayPreferencesNode->GetInt(DNDDISPLAY_DEFAULT_INTERPOLATION_TYPE, 2));
  ui->m_DefaultDropTypeComboBox->setCurrentIndex(m_DnDDisplayPreferencesNode->GetInt(DNDDISPLAY_DEFAULT_DROP_TYPE, 0));
  ui->m_ShowDropTypeControlsCheckBox->setChecked(m_DnDDisplayPreferencesNode->GetBool(DNDDISPLAY_SHOW_DROP_TYPE_CONTROLS, false));
  ui->m_ShowShowOptionsCheckBox->setChecked(m_DnDDisplayPreferencesNode->GetBool(DNDDISPLAY_SHOW_SHOWING_OPTIONS, true));
  ui->m_ShowWindowLayoutControlsCheckBox->setChecked(m_DnDDisplayPreferencesNode->GetBool(DNDDISPLAY_SHOW_WINDOW_LAYOUT_CONTROLS, true));
  ui->m_ShowViewerNumberControlsCheckBox->setChecked(m_DnDDisplayPreferencesNode->GetBool(DNDDISPLAY_SHOW_VIEWER_NUMBER_CONTROLS, true));
  ui->m_Show2DCursorsCheckBox->setChecked(m_DnDDisplayPreferencesNode->GetBool(DNDDISPLAY_SHOW_2D_CURSORS, true));
  ui->m_ShowDirectionAnnotationsCheckBox->setChecked(m_DnDDisplayPreferencesNode->GetBool(DNDDISPLAY_SHOW_DIRECTION_ANNOTATIONS, true));
  ui->m_ShowPositionAnnotationCheckBox->setChecked(m_DnDDisplayPreferencesNode->GetBool(DNDDISPLAY_SHOW_POSITION_ANNOTATION, true));
  ui->m_ShowIntensityAnnotationCheckBox->setChecked(m_DnDDisplayPreferencesNode->GetBool(DNDDISPLAY_SHOW_INTENSITY_ANNOTATION, true));
  ui->m_ShowPropertyAnnotationCheckBox->setChecked(m_DnDDisplayPreferencesNode->GetBool(DNDDISPLAY_SHOW_PROPERTY_ANNOTATION, false));
  ui->m_PropertiesForAnnotationLineEdit->setText(m_DnDDisplayPreferencesNode->Get(DNDDISPLAY_PROPERTIES_FOR_ANNOTATION, "name"));
  ui->m_ShowMagnificationSliderCheckBox->setChecked(m_DnDDisplayPreferencesNode->GetBool(DNDDISPLAY_SHOW_MAGNIFICATION_SLIDER, true));
  ui->m_RememberViewerSettingsPerWindowLayoutCheckBox->setChecked(m_DnDDisplayPreferencesNode->GetBool(DNDDISPLAY_REMEMBER_VIEWER_SETTINGS_PER_WINDOW_LAYOUT, true));
  ui->m_SliceTrackingCheckBox->setChecked(m_DnDDisplayPreferencesNode->GetBool(DNDDISPLAY_SLICE_SELECT_TRACKING, true));
  ui->m_MagnificationTrackingCheckBox->setChecked(m_DnDDisplayPreferencesNode->GetBool(DNDDISPLAY_MAGNIFICATION_SELECT_TRACKING, true));
  ui->m_TimeTrackingCheckBox->setChecked(m_DnDDisplayPreferencesNode->GetBool(DNDDISPLAY_TIME_SELECT_TRACKING, true));
}


//-----------------------------------------------------------------------------
void DnDDisplayPreferencePage::OnBackgroundColourChanged()
{
  QColor colour = QColorDialog::getColor();
  if (colour.isValid())
  {
    ui->m_BackgroundColourColourButton->setAutoFillBackground(true);

    QString styleSheet = "background-color: rgb(";
    styleSheet.append(QString::number(colour.red()));
    styleSheet.append(",");
    styleSheet.append(QString::number(colour.green()));
    styleSheet.append(",");
    styleSheet.append(QString::number(colour.blue()));
    styleSheet.append(")");

    ui->m_BackgroundColourColourButton->setStyleSheet(styleSheet);
    m_BackgroundColorStyleSheet = styleSheet;

    QStringList backgroundColour;
    backgroundColour << colour.name();

    m_BackgroundColor = backgroundColour.replaceInStrings(";","\\;").join(";");
  }
 }


//-----------------------------------------------------------------------------
void DnDDisplayPreferencePage::OnResetBackgroundColour()
{
  m_BackgroundColorStyleSheet = "background-color: rgb(0, 0, 0)";
  m_BackgroundColor = "#000000";
  ui->m_BackgroundColourColourButton->setStyleSheet(m_BackgroundColorStyleSheet);
}


//-----------------------------------------------------------------------------
void DnDDisplayPreferencePage::OnResetMIDASBackgroundColour()
{
  m_BackgroundColorStyleSheet = "background-color: rgb(255,250,240)"; // That strange MIDAS off-white colour.
  m_BackgroundColor = "#fffaf0";                                      // That strange MIDAS off-white colour.
  ui->m_BackgroundColourColourButton->setStyleSheet(m_BackgroundColorStyleSheet);
}

}
