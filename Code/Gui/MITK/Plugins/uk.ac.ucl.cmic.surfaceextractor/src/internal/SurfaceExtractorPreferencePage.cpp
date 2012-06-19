/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-28 10:00:55 +0100 (Wed, 28 Sep 2011) $
 Revision          : $Revision: 7379 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "SurfaceExtractorPreferencePage.h"

#include <QFormLayout>
#include <QLabel>
#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QSpinBox>

#include <berryIPreferencesService.h>
#include <berryPlatform.h>

#include <limits>

class SurfaceExtractorPreferencePagePrivate
{
public:
  bool initializing;

  berry::IPreferences::Pointer preferencesNode;

  QGridLayout *gridLayout;
  QLabel *lblRefImageLabel;
  QLabel *lblReferenceImage;
  QLabel *lblThreshold;
  QDoubleSpinBox *spbThreshold;
  QLabel *lblTargetReduction;
  QLabel *lblMaxNumberOfPolygons;
  QSpinBox *spbMaxNumberOfPolygons;
  QLabel *lblGaussianSmooth;
  QLabel *lblGaussianStdDev;
  QCheckBox *cbxGaussianSmooth;
  QDoubleSpinBox *spbGaussianStdDev;
  QDoubleSpinBox *spbTargetReduction;
};


const std::string SurfaceExtractorPreferencePage::GAUSSIAN_SMOOTH_NAME = "surface_extractor.gaussian_smooth";
const bool SurfaceExtractorPreferencePage::GAUSSIAN_SMOOTH_DEFAULT = true;
const std::string SurfaceExtractorPreferencePage::GAUSSIAN_STDDEV_NAME = "surface_extractor.gaussian_standard_deviation";
const double SurfaceExtractorPreferencePage::GAUSSIAN_STDDEV_DEFAULT = 0.5;
const std::string SurfaceExtractorPreferencePage::THRESHOLD_NAME = "surface_extractor.threshold";
const double SurfaceExtractorPreferencePage::THRESHOLD_DEFAULT = 100.0;
const std::string SurfaceExtractorPreferencePage::TARGET_REDUCTION_NAME = "surface_extractor.target_reduction";
const double SurfaceExtractorPreferencePage::TARGET_REDUCTION_DEFAULT = 0.1;
const std::string SurfaceExtractorPreferencePage::MAX_NUMBER_OF_POLYGONS_NAME = "surface_extractor.maximal_number_of_polygons";
const long SurfaceExtractorPreferencePage::MAX_NUMBER_OF_POLYGONS_DEFAULT = 2000000;

SurfaceExtractorPreferencePage::SurfaceExtractorPreferencePage()
: ui(0)
, d_ptr(new SurfaceExtractorPreferencePagePrivate())
{
  Q_D(SurfaceExtractorPreferencePage);
  d->initializing = false;

  berry::IPreferencesService::Pointer prefService
    = berry::Platform::GetServiceRegistry()
      .GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);

  d->preferencesNode = prefService->GetSystemPreferences()->Node("/uk.ac.ucl.cmic.SurfaceExtractor");
}

SurfaceExtractorPreferencePage::~SurfaceExtractorPreferencePage()
{
  if (ui)
  {
    delete ui;
  }
}

void SurfaceExtractorPreferencePage::Init(berry::IWorkbench::Pointer)
{
}

void SurfaceExtractorPreferencePage::CreateQtControl(QWidget* parent)
{
  Q_D(SurfaceExtractorPreferencePage);

  d->initializing = true;
  ui = new QWidget(parent);

  QFormLayout *formLayout = new QFormLayout;

  d->lblThreshold = new QLabel("Threshold:");
  QString ttpThreshold = "Threshold";
  d->lblThreshold->setToolTip(ttpThreshold);
  d->spbThreshold = new QDoubleSpinBox(ui);
  d->spbThreshold->setMinimum(0);
  d->spbThreshold->setMaximum(std::numeric_limits<int>::max());
  d->spbThreshold->setToolTip(ttpThreshold);
  formLayout->addRow(d->lblThreshold, d->spbThreshold);

  d->lblGaussianSmooth = new QLabel("Apply Gaussian smoothing:");
  QString ttpGaussianSmooth = "Apply Gaussian smoothing";
  d->lblGaussianSmooth->setToolTip(ttpGaussianSmooth);
  d->cbxGaussianSmooth = new QCheckBox(ui);
  d->cbxGaussianSmooth->setToolTip(ttpGaussianSmooth);
  formLayout->addRow(d->lblGaussianSmooth, d->cbxGaussianSmooth);
  
  d->lblGaussianStdDev = new QLabel("Standard deviation:");
  QString ttpGaussianStdDev = "Standard deviation for the Gaussian smoothing";
  d->lblGaussianStdDev->setToolTip(ttpGaussianStdDev);
  d->spbGaussianStdDev = new QDoubleSpinBox(ui);
  d->spbGaussianStdDev->setToolTip(ttpGaussianStdDev);
  formLayout->addRow(d->lblGaussianStdDev, d->spbGaussianStdDev);

  d->lblTargetReduction = new QLabel("Target reduction:");
  QString ttpTargetReduction = "Target reduction";
  d->lblTargetReduction->setToolTip(ttpTargetReduction);
  d->spbTargetReduction = new QDoubleSpinBox(ui);
  d->spbTargetReduction->setToolTip(ttpTargetReduction);
  formLayout->addRow(d->lblTargetReduction, d->spbTargetReduction);

  d->lblMaxNumberOfPolygons = new QLabel("Maximal number of polygons:");
  QString ttpMaxNumberOfPolygons = "Maximal number of polygons";
  d->lblMaxNumberOfPolygons->setToolTip(ttpMaxNumberOfPolygons);
  d->spbMaxNumberOfPolygons = new QSpinBox(ui);
  d->spbMaxNumberOfPolygons->setMinimum(1);
  d->spbMaxNumberOfPolygons->setMaximum(std::numeric_limits<int>::max());
  d->spbMaxNumberOfPolygons->setToolTip(ttpMaxNumberOfPolygons);
  formLayout->addRow(d->lblMaxNumberOfPolygons, d->spbMaxNumberOfPolygons);

  ui->setLayout(formLayout);

  Update();

  connect(d->cbxGaussianSmooth, SIGNAL(toggled(bool)), this, SLOT(on_cbxGaussianSmooth_toggled(bool)));

  d->initializing = false;
}

QWidget* SurfaceExtractorPreferencePage::GetQtControl() const
{
  return ui;
}

bool SurfaceExtractorPreferencePage::PerformOk()
{
  Q_D(SurfaceExtractorPreferencePage);

  bool gaussianSmooth = d->cbxGaussianSmooth->isChecked();
  double gaussianStdDev = d->spbGaussianStdDev->value();
  double threshold = d->spbThreshold->value();
  double targetReduction = d->spbTargetReduction->value();
  long maxNumberOfPolygons = d->spbMaxNumberOfPolygons->value();

  d->preferencesNode->PutBool(GAUSSIAN_SMOOTH_NAME, gaussianSmooth);
  d->preferencesNode->PutDouble(GAUSSIAN_STDDEV_NAME, gaussianStdDev);
  d->preferencesNode->PutDouble(THRESHOLD_NAME, threshold);
  d->preferencesNode->PutDouble(TARGET_REDUCTION_NAME, targetReduction);
  d->preferencesNode->PutLong(MAX_NUMBER_OF_POLYGONS_NAME, maxNumberOfPolygons);

  return true;
}

void SurfaceExtractorPreferencePage::PerformCancel()
{
}

void SurfaceExtractorPreferencePage::Update()
{
  Q_D(SurfaceExtractorPreferencePage);

  bool gaussianSmooth = d->preferencesNode->GetBool(GAUSSIAN_SMOOTH_NAME, GAUSSIAN_SMOOTH_DEFAULT);
  double gaussianStdDev = d->preferencesNode->GetDouble(GAUSSIAN_STDDEV_NAME, GAUSSIAN_STDDEV_DEFAULT);
  double threshold = d->preferencesNode->GetDouble(THRESHOLD_NAME, THRESHOLD_DEFAULT);
  double targetReduction = d->preferencesNode->GetDouble(TARGET_REDUCTION_NAME, TARGET_REDUCTION_DEFAULT);
  long maxNumberOfPolygons = d->preferencesNode->GetLong(MAX_NUMBER_OF_POLYGONS_NAME, MAX_NUMBER_OF_POLYGONS_DEFAULT);

  d->cbxGaussianSmooth->setChecked(gaussianSmooth);
  d->spbGaussianStdDev->setValue(gaussianStdDev);
  d->spbGaussianStdDev->setEnabled(gaussianSmooth);
  d->spbThreshold->setValue(threshold);
  d->spbTargetReduction->setValue(targetReduction);
  d->spbMaxNumberOfPolygons->setValue(maxNumberOfPolygons);
}

void SurfaceExtractorPreferencePage::on_cbxGaussianSmooth_toggled(bool checked)
{
  Q_D(SurfaceExtractorPreferencePage);
  d->spbGaussianStdDev->setEnabled(checked);
}
