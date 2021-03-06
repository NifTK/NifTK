/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "SurfaceRegViewPreferencePage.h"
#include "SurfaceRegView.h"

#include <QFormLayout>
#include <QVBoxLayout>
#include <QLabel>
#include <QCheckBox>
#include <QSpinBox>
#include <QMessageBox>
#include <QPushButton>

#include <berryIPreferencesService.h>
#include <berryPlatform.h>
#include <niftkICPBasedRegistration.h>

const QString SurfaceRegViewPreferencePage::MAXIMUM_NUMBER_OF_ITERATIONS("maximum iterations");
const QString SurfaceRegViewPreferencePage::MAXIMUM_NUMBER_OF_POINTS("maximum number of points");
const QString SurfaceRegViewPreferencePage::TLS_ITERATIONS("Trimmed Least Squares iterations (zero is OFF)");
const QString SurfaceRegViewPreferencePage::TLS_PERCENTAGE("Trimmed Least Squares percentage");

//-----------------------------------------------------------------------------
SurfaceRegViewPreferencePage::SurfaceRegViewPreferencePage()
: m_MainControl(0)
, m_MaximumIterations(0)
, m_MaximumPoints(0)
, m_TLSIterations(0)
, m_TLSPercentage(0)
, m_Initializing(false)
, m_SurfaceRegViewPreferencesNode(0)
{
}


//-----------------------------------------------------------------------------
SurfaceRegViewPreferencePage::SurfaceRegViewPreferencePage(const SurfaceRegViewPreferencePage& other)
: berry::Object(), QObject()
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
SurfaceRegViewPreferencePage::~SurfaceRegViewPreferencePage()
{
}


//-----------------------------------------------------------------------------
void SurfaceRegViewPreferencePage::Init(berry::IWorkbench::Pointer )
{

}


//-----------------------------------------------------------------------------
void SurfaceRegViewPreferencePage::CreateQtControl(QWidget* parent)
{
  m_Initializing = true;

  berry::IPreferencesService* prefService = berry::Platform::GetPreferencesService();
  m_SurfaceRegViewPreferencesNode = prefService->GetSystemPreferences()->Node(SurfaceRegView::VIEW_ID);

  m_MainControl = new QWidget(parent);
  QFormLayout *formLayout = new QFormLayout;

  m_MaximumIterations = new QSpinBox();
  m_MaximumIterations->setMinimum(0);
  m_MaximumIterations->setMaximum(2000);

  m_MaximumPoints = new QSpinBox();
  m_MaximumPoints->setMinimum (3);
  m_MaximumPoints->setMaximum (10000);

  m_TLSIterations = new QSpinBox();
  m_TLSIterations->setMinimum (0);
  m_TLSIterations->setMaximum (10);

  m_TLSPercentage = new QSpinBox();
  m_TLSPercentage->setMinimum (1);
  m_TLSPercentage->setMaximum (100);

  formLayout->addRow("Maximum number of ICP iterations", m_MaximumIterations);
  formLayout->addRow("Maximum number of points to use in ICP", m_MaximumPoints);
  formLayout->addRow(TLS_ITERATIONS, m_TLSIterations);
  formLayout->addRow(TLS_PERCENTAGE, m_TLSPercentage);

  m_MainControl->setLayout(formLayout);
  this->Update();

  m_Initializing = false;
}


//-----------------------------------------------------------------------------
QWidget* SurfaceRegViewPreferencePage::GetQtControl() const
{
  return m_MainControl;
}


//-----------------------------------------------------------------------------
bool SurfaceRegViewPreferencePage::PerformOk()
{
  m_SurfaceRegViewPreferencesNode->PutInt(SurfaceRegViewPreferencePage::MAXIMUM_NUMBER_OF_ITERATIONS, m_MaximumIterations->value());
  m_SurfaceRegViewPreferencesNode->PutInt(SurfaceRegViewPreferencePage::MAXIMUM_NUMBER_OF_POINTS, m_MaximumPoints->value());
  m_SurfaceRegViewPreferencesNode->PutInt(SurfaceRegViewPreferencePage::TLS_ITERATIONS, m_TLSIterations->value());
  m_SurfaceRegViewPreferencesNode->PutInt(SurfaceRegViewPreferencePage::TLS_PERCENTAGE, m_TLSPercentage->value());

  return true;
}


//-----------------------------------------------------------------------------
void SurfaceRegViewPreferencePage::PerformCancel()
{

}


//-----------------------------------------------------------------------------
void SurfaceRegViewPreferencePage::Update()
{
  m_MaximumIterations->setValue(m_SurfaceRegViewPreferencesNode->GetInt(SurfaceRegViewPreferencePage::MAXIMUM_NUMBER_OF_ITERATIONS, 
        niftk::ICPBasedRegistrationConstants::DEFAULT_MAX_ITERATIONS));
  m_MaximumPoints->setValue(m_SurfaceRegViewPreferencesNode->GetInt(SurfaceRegViewPreferencePage::MAXIMUM_NUMBER_OF_POINTS,
        niftk::ICPBasedRegistrationConstants::DEFAULT_MAX_POINTS));
  m_TLSIterations->setValue(m_SurfaceRegViewPreferencesNode->GetInt(SurfaceRegViewPreferencePage::TLS_ITERATIONS,
        niftk::ICPBasedRegistrationConstants::DEFAULT_TLS_ITERATIONS));
  m_TLSPercentage->setValue(m_SurfaceRegViewPreferencesNode->GetInt(SurfaceRegViewPreferencePage::TLS_PERCENTAGE,
        niftk::ICPBasedRegistrationConstants::DEFAULT_TLS_PERCENTAGE));

}
