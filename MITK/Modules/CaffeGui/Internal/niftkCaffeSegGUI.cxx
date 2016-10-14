/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkCaffeSegGUI.h"
#include <mitkNodePredicateDataType.h>
#include <mitkImage.h>

namespace niftk
{

//-----------------------------------------------------------------------------
CaffeSegGUI::CaffeSegGUI(QWidget* parent)
: BaseGUI(parent)
{
  this->setupUi(parent);
  this->m_ManualUpdateRadioButton->setChecked(false);
  this->m_AutomaticUpdateRadioButton->setChecked(true);
  this->connect(m_LeftImageComboBox, SIGNAL(OnSelectionChanged(const mitk::DataNode*)), SIGNAL(OnLeftSelectionChanged(const mitk::DataNode*)));
  this->connect(m_RightImageComboBox, SIGNAL(OnSelectionChanged(const mitk::DataNode*)), SIGNAL(OnRightSelectionChanged(const mitk::DataNode*)));
  this->connect(m_ManualUpdatePushButton, SIGNAL(pressed()), SIGNAL(OnDoItNowPressed()));
  this->connect(m_ManualUpdateRadioButton, SIGNAL(clicked(bool)), SIGNAL(OnManualUpdateClicked(bool)));
  this->connect(m_AutomaticUpdateRadioButton, SIGNAL(clicked(bool)), SIGNAL(OnAutomaticUpdateClicked(bool)));
}


//-----------------------------------------------------------------------------
CaffeSegGUI::~CaffeSegGUI()
{

}


//-----------------------------------------------------------------------------
void CaffeSegGUI::SetDataStorage(mitk::DataStorage* storage)
{
  mitk::TNodePredicateDataType<mitk::Image>::Pointer isImage = mitk::TNodePredicateDataType<mitk::Image>::New();
  m_LeftImageComboBox->SetAutoSelectNewItems(false);
  m_LeftImageComboBox->SetPredicate(isImage);
  m_LeftImageComboBox->SetDataStorage(storage);
  m_LeftImageComboBox->setCurrentIndex(0);

  m_RightImageComboBox->SetAutoSelectNewItems(false);
  m_RightImageComboBox->SetPredicate(isImage);
  m_RightImageComboBox->SetDataStorage(storage);
  m_RightImageComboBox->setCurrentIndex(0);
}

} // end namespace
