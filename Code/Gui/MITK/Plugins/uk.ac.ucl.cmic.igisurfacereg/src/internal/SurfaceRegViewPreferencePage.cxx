/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "SurfaceRegViewPreferencePage.h"

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

const QString SurfaceRegViewPreferencePage::PREFERENCES_NODE_NAME("/uk.ac.ucl.cmic.igisurfacereg");

//-----------------------------------------------------------------------------
SurfaceRegViewPreferencePage::SurfaceRegViewPreferencePage()
: m_MainControl(0)
, m_MaximumIterations(0)
, m_MaximumPoints(0)
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

  m_SurfaceRegViewPreferencesNode = prefService->GetSystemPreferences()->Node(PREFERENCES_NODE_NAME);

  m_MainControl = new QWidget(parent);
  QFormLayout *formLayout = new QFormLayout;

  m_MaximumIterations = new QSpinBox();
  m_MaximumIterations->setMinimum(0);
  m_MaximumIterations->setMaximum(2000);

  m_MaximumPoints = new QSpinBox();
  m_MaximumPoints->setMinimum (3);
  m_MaximumPoints->setMaximum (10000);

  formLayout->addRow("Maximum number of ICP iterations", m_MaximumIterations);
  formLayout->addRow("Maximum number of points to use in ICP", m_MaximumPoints);

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
  m_SurfaceRegViewPreferencesNode->PutInt("Maximum number of ICP iterations",m_MaximumIterations->value());
  m_SurfaceRegViewPreferencesNode->PutInt("Maximum number of points to use in ICP",m_MaximumPoints->value());
  return true;
}


//-----------------------------------------------------------------------------
void SurfaceRegViewPreferencePage::PerformCancel()
{

}


//-----------------------------------------------------------------------------
void SurfaceRegViewPreferencePage::Update()
{
  m_MaximumIterations->setValue(m_SurfaceRegViewPreferencesNode->GetInt("Maximum number of ICP iterations",niftk::ICPBasedRegistrationConstants::DEFAULT_MAX_ITERATIONS));
  m_MaximumPoints->setValue(m_SurfaceRegViewPreferencesNode->GetInt("Maximum number of points to use in ICP",niftk::ICPBasedRegistrationConstants::DEFAULT_MAX_POINTS));
}
