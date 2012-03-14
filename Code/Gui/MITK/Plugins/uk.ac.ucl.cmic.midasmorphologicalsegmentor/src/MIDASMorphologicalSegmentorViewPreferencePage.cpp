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

#include "MIDASMorphologicalSegmentorViewPreferencePage.h"

#include <QFormLayout>
#include <QVBoxLayout>
#include <QLabel>
#include <QCheckBox>
#include <QMessageBox>

#include <berryIPreferencesService.h>
#include <berryPlatform.h>

const std::string MIDASMorphologicalSegmentorViewPreferencePage::MIDAS_MORPH_DO_VOLUME_RENDERING("midas morphological editor do volume rendering");

MIDASMorphologicalSegmentorViewPreferencePage::MIDASMorphologicalSegmentorViewPreferencePage()
: m_MainControl(0)
, m_DoVolumeRenderingCheckBox(0)
, m_Initializing(false)
{

}

MIDASMorphologicalSegmentorViewPreferencePage::MIDASMorphologicalSegmentorViewPreferencePage(const MIDASMorphologicalSegmentorViewPreferencePage& other)
: berry::Object(), QObject()
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}

MIDASMorphologicalSegmentorViewPreferencePage::~MIDASMorphologicalSegmentorViewPreferencePage()
{

}

void MIDASMorphologicalSegmentorViewPreferencePage::Init(berry::IWorkbench::Pointer )
{

}

void MIDASMorphologicalSegmentorViewPreferencePage::CreateQtControl(QWidget* parent)
{
  m_Initializing = true;
  berry::IPreferencesService::Pointer prefService
    = berry::Platform::GetServiceRegistry()
      .GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);

  m_MIDASMorphologicalSegmentorViewPreferencesNode = prefService->GetSystemPreferences()->Node("/uk_ac_ucl_cmic_midasmorphologicalsegmentor");

  m_MainControl = new QWidget(parent);

  QFormLayout *formLayout = new QFormLayout;

  m_DoVolumeRenderingCheckBox = new QCheckBox(parent);
  formLayout->addRow("do volume rendering", m_DoVolumeRenderingCheckBox);

  m_MainControl->setLayout(formLayout);
  this->Update();

  m_Initializing = false;
}

QWidget* MIDASMorphologicalSegmentorViewPreferencePage::GetQtControl() const
{
  return m_MainControl;
}

bool MIDASMorphologicalSegmentorViewPreferencePage::PerformOk()
{
  m_MIDASMorphologicalSegmentorViewPreferencesNode->PutBool(MIDAS_MORPH_DO_VOLUME_RENDERING, m_DoVolumeRenderingCheckBox->isChecked());
  return true;
}

void MIDASMorphologicalSegmentorViewPreferencePage::PerformCancel()
{

}

void MIDASMorphologicalSegmentorViewPreferencePage::Update()
{
  if (m_MIDASMorphologicalSegmentorViewPreferencesNode->GetBool(MIDAS_MORPH_DO_VOLUME_RENDERING, false) )
  {
    m_DoVolumeRenderingCheckBox->setChecked( true );
  }
  else
  {
    m_DoVolumeRenderingCheckBox->setChecked( false );
  }
}
