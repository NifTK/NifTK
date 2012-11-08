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

#include "QmitkImageLookupTablesPreferencePage.h"

#include <QFormLayout>
#include <QVBoxLayout>
#include <QLabel>
#include <QSpinBox>

#include <berryIPreferencesService.h>
#include <berryPlatform.h>

const std::string QmitkImageLookupTablesPreferencePage::PRECISION_NAME("precision");

//-----------------------------------------------------------------------------
QmitkImageLookupTablesPreferencePage::QmitkImageLookupTablesPreferencePage()
: m_MainControl(0)
, m_Precision(0)
, m_Initializing(false)
{

}


//-----------------------------------------------------------------------------
QmitkImageLookupTablesPreferencePage::QmitkImageLookupTablesPreferencePage(const QmitkImageLookupTablesPreferencePage& other)
: berry::Object(), QObject()
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
QmitkImageLookupTablesPreferencePage::~QmitkImageLookupTablesPreferencePage()
{

}


//-----------------------------------------------------------------------------
void QmitkImageLookupTablesPreferencePage::Init(berry::IWorkbench::Pointer )
{

}


//-----------------------------------------------------------------------------
void QmitkImageLookupTablesPreferencePage::CreateQtControl(QWidget* parent)
{
  m_Initializing = true;
  berry::IPreferencesService::Pointer prefService 
    = berry::Platform::GetServiceRegistry()
      .GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);

  m_ImageLookupTablesPreferencesNode = prefService->GetSystemPreferences()->Node("/uk.ac.ucl.cmic.imagelookuptables");

  m_MainControl = new QWidget(parent);

  QFormLayout *formLayout = new QFormLayout;

  QLabel* precisionLabel = new QLabel("Precision:");
  m_Precision = new QSpinBox;
  QString precisionToolTip =
      "Precision of the floating point numbers.";
  precisionLabel->setToolTip(precisionToolTip);
  m_Precision->setToolTip(precisionToolTip);
  formLayout->addRow(precisionLabel, m_Precision);

  m_MainControl->setLayout(formLayout);
  this->Update();

  m_Initializing = false;
}


//-----------------------------------------------------------------------------
QWidget* QmitkImageLookupTablesPreferencePage::GetQtControl() const
{
  return m_MainControl;
}


//-----------------------------------------------------------------------------
bool QmitkImageLookupTablesPreferencePage::PerformOk()
{
  m_ImageLookupTablesPreferencesNode->PutInt(PRECISION_NAME, m_Precision->text().toInt());
  return true;
}


//-----------------------------------------------------------------------------
void QmitkImageLookupTablesPreferencePage::PerformCancel()
{

}


//-----------------------------------------------------------------------------
void QmitkImageLookupTablesPreferencePage::Update()
{
  m_Precision->setValue(m_ImageLookupTablesPreferencesNode->GetInt(PRECISION_NAME, 2));
}
