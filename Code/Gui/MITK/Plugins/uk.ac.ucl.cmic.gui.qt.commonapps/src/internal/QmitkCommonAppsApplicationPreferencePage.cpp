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

#include "QmitkCommonAppsApplicationPreferencePage.h"

#include <QFormLayout>
#include <QVBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QComboBox>
#include <QCheckBox>

#include <berryIPreferencesService.h>
#include <berryPlatform.h>

const std::string QmitkCommonAppsApplicationPreferencePage::IMAGE_RESLICE_INTERPOLATION("default reslice interpolation");
const std::string QmitkCommonAppsApplicationPreferencePage::IMAGE_TEXTURE_INTERPOLATION("default texture interpolation");
const std::string QmitkCommonAppsApplicationPreferencePage::BLACK_OPACITY("black opacity");

//-----------------------------------------------------------------------------
QmitkCommonAppsApplicationPreferencePage::QmitkCommonAppsApplicationPreferencePage()
: m_MainControl(0)
, m_ResliceInterpolation(0)
, m_TextureInterpolation(0)
, m_BlackOpacity(0)
{

}


//-----------------------------------------------------------------------------
QmitkCommonAppsApplicationPreferencePage::QmitkCommonAppsApplicationPreferencePage(const QmitkCommonAppsApplicationPreferencePage& other)
: berry::Object(), QObject()
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
QmitkCommonAppsApplicationPreferencePage::~QmitkCommonAppsApplicationPreferencePage()
{

}


//-----------------------------------------------------------------------------
void QmitkCommonAppsApplicationPreferencePage::Init(berry::IWorkbench::Pointer )
{

}


//-----------------------------------------------------------------------------
void QmitkCommonAppsApplicationPreferencePage::CreateQtControl(QWidget* parent)
{
  berry::IPreferencesService::Pointer prefService
    = berry::Platform::GetServiceRegistry()
      .GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);

  m_PreferencesNode = prefService->GetSystemPreferences()->Node("/uk.ac.ucl.cmic.gui.qt.commonapps");

  m_MainControl = new QWidget(parent);

  m_ResliceInterpolation = new QComboBox();
  m_ResliceInterpolation->insertItem(0, "none");
  m_ResliceInterpolation->insertItem(1, "linear");
  m_ResliceInterpolation->insertItem(2, "cubic");

  m_TextureInterpolation = new QComboBox();
  m_TextureInterpolation->insertItem(0, "none");
  m_TextureInterpolation->insertItem(1, "linear");

  m_BlackOpacity = new QCheckBox();

  QFormLayout *formLayout = new QFormLayout;
  formLayout->addRow( "Image reslice interpolation:", m_ResliceInterpolation );
  formLayout->addRow( "Image texture interpolation:", m_TextureInterpolation );
  formLayout->addRow( "Opacity of black (lowest value in lookup table):", m_BlackOpacity);

  m_MainControl->setLayout(formLayout);
  this->Update();
}


//-----------------------------------------------------------------------------
QWidget* QmitkCommonAppsApplicationPreferencePage::GetQtControl() const
{
  return m_MainControl;
}


//-----------------------------------------------------------------------------
bool QmitkCommonAppsApplicationPreferencePage::PerformOk()
{
  m_PreferencesNode->PutInt(IMAGE_RESLICE_INTERPOLATION, m_ResliceInterpolation->currentIndex());
  m_PreferencesNode->PutInt(IMAGE_TEXTURE_INTERPOLATION, m_TextureInterpolation->currentIndex());
  m_PreferencesNode->PutBool(BLACK_OPACITY, m_BlackOpacity->isChecked());
  return true;
}


//-----------------------------------------------------------------------------
void QmitkCommonAppsApplicationPreferencePage::PerformCancel()
{

}


//-----------------------------------------------------------------------------
void QmitkCommonAppsApplicationPreferencePage::Update()
{
  m_ResliceInterpolation->setCurrentIndex(m_PreferencesNode->GetInt(IMAGE_RESLICE_INTERPOLATION, 2));
  m_TextureInterpolation->setCurrentIndex(m_PreferencesNode->GetInt(IMAGE_TEXTURE_INTERPOLATION, 1));
  m_BlackOpacity->setChecked(m_PreferencesNode->GetBool(BLACK_OPACITY, true));
}
