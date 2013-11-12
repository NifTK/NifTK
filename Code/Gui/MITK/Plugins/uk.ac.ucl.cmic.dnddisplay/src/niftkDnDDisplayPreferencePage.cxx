/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkDnDDisplayPreferencePage.h"
#include "niftkDnDDisplayEditor.h"

#include <QLabel>
#include <QPushButton>
#include <QFormLayout>
#include <QColorDialog>
#include <QComboBox>
#include <QSpinBox>
#include <QCheckBox>
#include <berryIPreferencesService.h>
#include <berryPlatform.h>

const std::string niftkDnDDisplayPreferencePage::DNDDISPLAY_DEFAULT_INTERPOLATION_TYPE("default image interpolation");
const std::string niftkDnDDisplayPreferencePage::DNDDISPLAY_BACKGROUND_COLOUR("DnD display background colour");
const std::string niftkDnDDisplayPreferencePage::DNDDISPLAY_BACKGROUND_COLOUR_STYLESHEET("DnD display background colour stylesheet");

const std::string niftkDnDDisplayPreferencePage::DNDDISPLAY_SLICE_SELECT_TRACKING("DnD display slice select tracking");
const std::string niftkDnDDisplayPreferencePage::DNDDISPLAY_TIME_SELECT_TRACKING("DnD display time select tracking");
const std::string niftkDnDDisplayPreferencePage::DNDDISPLAY_MAGNIFICATION_SELECT_TRACKING("DnD display magnification select tracking");

const std::string niftkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_2D_CURSORS("DnD display show 2D cursors");
const std::string niftkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_DIRECTION_ANNOTATIONS("DnD display show direction annotations");
const std::string niftkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_3D_WINDOW_IN_MULTI_WINDOW_LAYOUT("DnD display show 3D window in multiple window layout");

const std::string niftkDnDDisplayPreferencePage::DNDDISPLAY_DEFAULT_WINDOW_LAYOUT("DnD display default window layout");
const std::string niftkDnDDisplayPreferencePage::DNDDISPLAY_REMEMBER_VIEWER_SETTINGS_PER_WINDOW_LAYOUT("DnD display remember view settings of each window layout");

const std::string niftkDnDDisplayPreferencePage::DNDDISPLAY_DEFAULT_VIEWER_ROW_NUMBER("DnD display default number of view rows");
const std::string niftkDnDDisplayPreferencePage::DNDDISPLAY_DEFAULT_VIEWER_COLUMN_NUMBER("DnD display default number of view columns");

const std::string niftkDnDDisplayPreferencePage::DNDDISPLAY_DEFAULT_DROP_TYPE("DnD display default drop type");

const std::string niftkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_MAGNIFICATION_SLIDER("DnD display show magnification slider");
const std::string niftkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_SHOWING_OPTIONS("DnD display show showing options");
const std::string niftkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_WINDOW_LAYOUT_CONTROLS("DnD display show window layout controls");
const std::string niftkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_VIEWER_NUMBER_CONTROLS("DnD display show view number controls");
const std::string niftkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_DROP_TYPE_CONTROLS("DnD display show drop type widgets");

//-----------------------------------------------------------------------------
niftkDnDDisplayPreferencePage::niftkDnDDisplayPreferencePage()
: m_MainControl(0)
, m_ImageInterpolationComboBox(NULL)
, m_SliceSelectTracking(NULL)
, m_TimeSelectTracking(NULL)
, m_MagnificationSelectTracking(NULL)
, m_Show2DCursorsCheckBox(NULL)
, m_ShowDirectionAnnotationsCheckBox(NULL)
, m_Show3DWindowInMultiWindowLayoutCheckBox(NULL)
, m_DefaultWindowLayoutComboBox(NULL)
, m_RememberEachWindowLayoutsViewerSettings(NULL)
, m_DefaultNumberOfViewerRowsSpinBox(NULL)
, m_DefaultNumberOfViewerColumnsSpinBox(NULL)
, m_DefaultDropType(NULL)
, m_ShowMagnificationSliderCheckBox(NULL)
, m_ShowShowingOptionsCheckBox(NULL)
, m_ShowWindowLayoutControlsCheckBox(NULL)
, m_ShowViewerNumberControlsCheckBox(NULL)
, m_ShowDropTypeControlsCheckBox(NULL)
, m_BackgroundColourButton(NULL)
{
}


//-----------------------------------------------------------------------------
niftkDnDDisplayPreferencePage::niftkDnDDisplayPreferencePage(const niftkDnDDisplayPreferencePage& other)
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
void niftkDnDDisplayPreferencePage::Init(berry::IWorkbench::Pointer )
{
}


//-----------------------------------------------------------------------------
void niftkDnDDisplayPreferencePage::CreateQtControl(QWidget* parent)
{
  berry::IPreferencesService::Pointer prefService
    = berry::Platform::GetServiceRegistry()
    .GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);

  m_DnDDisplayPreferencesNode = prefService->GetSystemPreferences()->Node(niftkDnDDisplayEditor::EDITOR_ID);

  m_MainControl = new QWidget(parent);

  QFormLayout* formLayout = new QFormLayout;

  m_ImageInterpolationComboBox = new QComboBox(parent);
  m_ImageInterpolationComboBox->insertItem(0, "none");
  m_ImageInterpolationComboBox->insertItem(1, "linear");
  m_ImageInterpolationComboBox->insertItem(2, "cubic");
  formLayout->addRow("image interpolation", m_ImageInterpolationComboBox);

  m_SliceSelectTracking = new QCheckBox(parent);
  formLayout->addRow("slice select tracking", m_SliceSelectTracking);

  m_TimeSelectTracking = new QCheckBox(parent);
  formLayout->addRow("time select tracking", m_TimeSelectTracking);

  m_MagnificationSelectTracking = new QCheckBox(parent);
  formLayout->addRow("magnification select tracking", m_MagnificationSelectTracking);

  m_Show2DCursorsCheckBox = new QCheckBox(parent);
  formLayout->addRow("show 2D cursors", m_Show2DCursorsCheckBox);

  m_ShowDirectionAnnotationsCheckBox = new QCheckBox(parent);
  formLayout->addRow("show direction annotations", m_ShowDirectionAnnotationsCheckBox);

  m_Show3DWindowInMultiWindowLayoutCheckBox = new QCheckBox(parent);
  formLayout->addRow("show 3D window in multiple window layout", m_Show3DWindowInMultiWindowLayoutCheckBox);

  m_DefaultWindowLayoutComboBox = new QComboBox(parent);
  m_DefaultWindowLayoutComboBox->insertItem(0, "axial");
  m_DefaultWindowLayoutComboBox->insertItem(1, "sagittal");
  m_DefaultWindowLayoutComboBox->insertItem(2, "coronal");
  m_DefaultWindowLayoutComboBox->insertItem(3, "2x2 orthogonal");
  m_DefaultWindowLayoutComboBox->insertItem(4, "3D");
  m_DefaultWindowLayoutComboBox->insertItem(5, "3 horizontal");
  m_DefaultWindowLayoutComboBox->insertItem(6, "3 vertical");
  m_DefaultWindowLayoutComboBox->insertItem(7, "as acquired (XY plane)");
  formLayout->addRow("default window layout", m_DefaultWindowLayoutComboBox);

  m_RememberEachWindowLayoutsViewerSettings = new QCheckBox(parent);
  formLayout->addRow("remember settings of each window layout", m_RememberEachWindowLayoutsViewerSettings);

  m_DefaultNumberOfViewerRowsSpinBox = new QSpinBox(parent);
  m_DefaultNumberOfViewerRowsSpinBox->setMinimum(1);
  m_DefaultNumberOfViewerRowsSpinBox->setMaximum(5);
  formLayout->addRow("initial number of view rows", m_DefaultNumberOfViewerRowsSpinBox);

  m_DefaultNumberOfViewerColumnsSpinBox = new QSpinBox(parent);
  m_DefaultNumberOfViewerColumnsSpinBox->setMinimum(1);
  m_DefaultNumberOfViewerColumnsSpinBox->setMaximum(5);
  formLayout->addRow("initial number of view columns", m_DefaultNumberOfViewerColumnsSpinBox);

  m_DefaultDropType = new QComboBox(parent);
  m_DefaultDropType->insertItem(0, "single");
  m_DefaultDropType->insertItem(1, "multiple");
  m_DefaultDropType->insertItem(2, "all");
  formLayout->addRow("default drop type", m_DefaultDropType);

  m_ShowMagnificationSliderCheckBox = new QCheckBox(parent);
  formLayout->addRow("show magnification slider", m_ShowMagnificationSliderCheckBox);

  m_ShowShowingOptionsCheckBox = new QCheckBox(parent);
  formLayout->addRow("show 'show' options", m_ShowShowingOptionsCheckBox);

  m_ShowWindowLayoutControlsCheckBox = new QCheckBox(parent);
  formLayout->addRow("show window layout controls", m_ShowWindowLayoutControlsCheckBox);

  m_ShowViewerNumberControlsCheckBox = new QCheckBox(parent);
  formLayout->addRow("show viewer number controls", m_ShowViewerNumberControlsCheckBox);

  m_ShowDropTypeControlsCheckBox = new QCheckBox(parent);
  formLayout->addRow("show drop type controls", m_ShowDropTypeControlsCheckBox);

  QPushButton* backgroundColourResetButton = new QPushButton(parent);
  backgroundColourResetButton->setText("reset");

  QPushButton* backgroundColorSpecificallyMIDAS = new QPushButton(parent);
  backgroundColorSpecificallyMIDAS->setText("MIDAS default");

  m_BackgroundColourButton = new QPushButton;
  m_BackgroundColourButton->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Minimum);

  QGridLayout* backgroundColourWidgetLayout = new QGridLayout;
  backgroundColourWidgetLayout->setContentsMargins(4, 4, 4, 4);
  backgroundColourWidgetLayout->addWidget(m_BackgroundColourButton, 0, 0);
  backgroundColourWidgetLayout->addWidget(backgroundColorSpecificallyMIDAS, 0, 1);
  backgroundColourWidgetLayout->addWidget(backgroundColourResetButton, 0, 2);

  formLayout->addRow("background colour", backgroundColourWidgetLayout);

  m_MainControl->setLayout(formLayout);

  this->connect( m_BackgroundColourButton, SIGNAL( clicked() ), SLOT( OnBackgroundColourChanged() ) );
  this->connect( backgroundColourResetButton, SIGNAL( clicked() ), SLOT( OnResetBackgroundColour() ) );
  this->connect( backgroundColorSpecificallyMIDAS, SIGNAL( clicked() ), SLOT( OnResetMIDASBackgroundColour() ) );

  this->Update();
}


//-----------------------------------------------------------------------------
QWidget* niftkDnDDisplayPreferencePage::GetQtControl() const
{
  return m_MainControl;
}


//-----------------------------------------------------------------------------
bool niftkDnDDisplayPreferencePage::PerformOk()
{
  m_DnDDisplayPreferencesNode->Put(DNDDISPLAY_BACKGROUND_COLOUR_STYLESHEET, m_BackgroundColorStyleSheet.toStdString());
  m_DnDDisplayPreferencesNode->PutByteArray(DNDDISPLAY_BACKGROUND_COLOUR, m_BackgroundColor);
  m_DnDDisplayPreferencesNode->PutInt(DNDDISPLAY_DEFAULT_VIEWER_ROW_NUMBER, m_DefaultNumberOfViewerRowsSpinBox->value());
  m_DnDDisplayPreferencesNode->PutInt(DNDDISPLAY_DEFAULT_VIEWER_COLUMN_NUMBER, m_DefaultNumberOfViewerColumnsSpinBox->value());
  m_DnDDisplayPreferencesNode->PutInt(DNDDISPLAY_DEFAULT_WINDOW_LAYOUT, m_DefaultWindowLayoutComboBox->currentIndex());
  m_DnDDisplayPreferencesNode->PutInt(DNDDISPLAY_DEFAULT_INTERPOLATION_TYPE, m_ImageInterpolationComboBox->currentIndex());
  m_DnDDisplayPreferencesNode->PutInt(DNDDISPLAY_DEFAULT_DROP_TYPE, m_DefaultDropType->currentIndex());
  m_DnDDisplayPreferencesNode->PutBool(DNDDISPLAY_SHOW_DROP_TYPE_CONTROLS, m_ShowDropTypeControlsCheckBox->isChecked());
  m_DnDDisplayPreferencesNode->PutBool(DNDDISPLAY_SHOW_SHOWING_OPTIONS, m_ShowShowingOptionsCheckBox->isChecked());
  m_DnDDisplayPreferencesNode->PutBool(DNDDISPLAY_SHOW_WINDOW_LAYOUT_CONTROLS, m_ShowWindowLayoutControlsCheckBox->isChecked());
  m_DnDDisplayPreferencesNode->PutBool(DNDDISPLAY_SHOW_VIEWER_NUMBER_CONTROLS, m_ShowViewerNumberControlsCheckBox->isChecked());
  m_DnDDisplayPreferencesNode->PutBool(DNDDISPLAY_SHOW_2D_CURSORS, m_Show2DCursorsCheckBox->isChecked());
  m_DnDDisplayPreferencesNode->PutBool(DNDDISPLAY_SHOW_DIRECTION_ANNOTATIONS, m_ShowDirectionAnnotationsCheckBox->isChecked());
  m_DnDDisplayPreferencesNode->PutBool(DNDDISPLAY_SHOW_3D_WINDOW_IN_MULTI_WINDOW_LAYOUT, m_Show3DWindowInMultiWindowLayoutCheckBox->isChecked());
  m_DnDDisplayPreferencesNode->PutBool(DNDDISPLAY_SHOW_MAGNIFICATION_SLIDER, m_ShowMagnificationSliderCheckBox->isChecked());
  m_DnDDisplayPreferencesNode->PutBool(DNDDISPLAY_REMEMBER_VIEWER_SETTINGS_PER_WINDOW_LAYOUT, m_RememberEachWindowLayoutsViewerSettings->isChecked());
  m_DnDDisplayPreferencesNode->PutBool(DNDDISPLAY_SLICE_SELECT_TRACKING, m_SliceSelectTracking->isChecked());
  m_DnDDisplayPreferencesNode->PutBool(DNDDISPLAY_MAGNIFICATION_SELECT_TRACKING, m_MagnificationSelectTracking->isChecked());
  m_DnDDisplayPreferencesNode->PutBool(DNDDISPLAY_TIME_SELECT_TRACKING, m_TimeSelectTracking->isChecked());
  return true;
}


//-----------------------------------------------------------------------------
void niftkDnDDisplayPreferencePage::PerformCancel()
{
}


//-----------------------------------------------------------------------------
void niftkDnDDisplayPreferencePage::Update()
{
  m_BackgroundColorStyleSheet = QString::fromStdString(m_DnDDisplayPreferencesNode->Get(DNDDISPLAY_BACKGROUND_COLOUR_STYLESHEET, ""));
  m_BackgroundColor = m_DnDDisplayPreferencesNode->GetByteArray(DNDDISPLAY_BACKGROUND_COLOUR, "");
  if (m_BackgroundColorStyleSheet=="")
  {
    m_BackgroundColorStyleSheet = "background-color: rgb(0, 0, 0)";
  }
  if (m_BackgroundColor=="")
  {
    m_BackgroundColor = "#000000";
  }
  m_BackgroundColourButton->setStyleSheet(m_BackgroundColorStyleSheet);

  m_DefaultNumberOfViewerRowsSpinBox->setValue(m_DnDDisplayPreferencesNode->GetInt(DNDDISPLAY_DEFAULT_VIEWER_ROW_NUMBER, 1));
  m_DefaultNumberOfViewerColumnsSpinBox->setValue(m_DnDDisplayPreferencesNode->GetInt(DNDDISPLAY_DEFAULT_VIEWER_COLUMN_NUMBER, 1));
  m_DefaultWindowLayoutComboBox->setCurrentIndex(m_DnDDisplayPreferencesNode->GetInt(DNDDISPLAY_DEFAULT_WINDOW_LAYOUT, 2)); // default coronal
  m_ImageInterpolationComboBox->setCurrentIndex(m_DnDDisplayPreferencesNode->GetInt(DNDDISPLAY_DEFAULT_INTERPOLATION_TYPE, 2));
  m_DefaultDropType->setCurrentIndex(m_DnDDisplayPreferencesNode->GetInt(DNDDISPLAY_DEFAULT_DROP_TYPE, 0));
  m_ShowDropTypeControlsCheckBox->setChecked(m_DnDDisplayPreferencesNode->GetBool(DNDDISPLAY_SHOW_DROP_TYPE_CONTROLS, false));
  m_ShowShowingOptionsCheckBox->setChecked(m_DnDDisplayPreferencesNode->GetBool(DNDDISPLAY_SHOW_SHOWING_OPTIONS, true));
  m_ShowWindowLayoutControlsCheckBox->setChecked(m_DnDDisplayPreferencesNode->GetBool(DNDDISPLAY_SHOW_WINDOW_LAYOUT_CONTROLS, true));
  m_ShowViewerNumberControlsCheckBox->setChecked(m_DnDDisplayPreferencesNode->GetBool(DNDDISPLAY_SHOW_VIEWER_NUMBER_CONTROLS, true));
  m_Show2DCursorsCheckBox->setChecked(m_DnDDisplayPreferencesNode->GetBool(DNDDISPLAY_SHOW_2D_CURSORS, true));
  m_ShowDirectionAnnotationsCheckBox->setChecked(m_DnDDisplayPreferencesNode->GetBool(DNDDISPLAY_SHOW_DIRECTION_ANNOTATIONS, true));
  m_Show3DWindowInMultiWindowLayoutCheckBox->setChecked(m_DnDDisplayPreferencesNode->GetBool(DNDDISPLAY_SHOW_3D_WINDOW_IN_MULTI_WINDOW_LAYOUT, false));
  m_ShowMagnificationSliderCheckBox->setChecked(m_DnDDisplayPreferencesNode->GetBool(DNDDISPLAY_SHOW_MAGNIFICATION_SLIDER, true));
  m_RememberEachWindowLayoutsViewerSettings->setChecked(m_DnDDisplayPreferencesNode->GetBool(DNDDISPLAY_REMEMBER_VIEWER_SETTINGS_PER_WINDOW_LAYOUT, true));
  m_SliceSelectTracking->setChecked(m_DnDDisplayPreferencesNode->GetBool(DNDDISPLAY_SLICE_SELECT_TRACKING, true));
  m_MagnificationSelectTracking->setChecked(m_DnDDisplayPreferencesNode->GetBool(DNDDISPLAY_MAGNIFICATION_SELECT_TRACKING, true));
  m_TimeSelectTracking->setChecked(m_DnDDisplayPreferencesNode->GetBool(DNDDISPLAY_TIME_SELECT_TRACKING, true));
}


//-----------------------------------------------------------------------------
void niftkDnDDisplayPreferencePage::OnBackgroundColourChanged()
{
  QColor colour = QColorDialog::getColor();
  if (colour.isValid())
  {
    m_BackgroundColourButton->setAutoFillBackground(true);

    QString styleSheet = "background-color: rgb(";
    styleSheet.append(QString::number(colour.red()));
    styleSheet.append(",");
    styleSheet.append(QString::number(colour.green()));
    styleSheet.append(",");
    styleSheet.append(QString::number(colour.blue()));
    styleSheet.append(")");

    m_BackgroundColourButton->setStyleSheet(styleSheet);
    m_BackgroundColorStyleSheet = styleSheet;

    QStringList backgroundColour;
    backgroundColour << colour.name();

    m_BackgroundColor = backgroundColour.replaceInStrings(";","\\;").join(";").toStdString();
  }
 }


//-----------------------------------------------------------------------------
void niftkDnDDisplayPreferencePage::OnResetBackgroundColour()
{
  m_BackgroundColorStyleSheet = "background-color: rgb(0, 0, 0)";
  m_BackgroundColor = "#000000";
  m_BackgroundColourButton->setStyleSheet(m_BackgroundColorStyleSheet);
}


//-----------------------------------------------------------------------------
void niftkDnDDisplayPreferencePage::OnResetMIDASBackgroundColour()
{
  m_BackgroundColorStyleSheet = "background-color: rgb(255,250,240)"; // That strange MIDAS off-white colour.
  m_BackgroundColor = "#fffaf0";                                      // That strange MIDAS off-white colour.
  m_BackgroundColourButton->setStyleSheet(m_BackgroundColorStyleSheet);
}
