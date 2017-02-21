/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkUSReconGUI.h"
#include <mitkNodePredicateDataType.h>
#include <mitkImage.h>
#include <niftkCoordinateAxesData.h>

namespace niftk
{

//-----------------------------------------------------------------------------
USReconGUI::USReconGUI(QWidget* parent)
: BaseGUI(parent)
{
  this->setupUi(parent);
  this->connect(m_ImageComboBox, SIGNAL(OnSelectionChanged(const mitk::DataNode*)), SIGNAL(OnImageSelectionChanged(const mitk::DataNode*)));
  this->connect(m_TrackingComboBox, SIGNAL(OnSelectionChanged(const mitk::DataNode*)), SIGNAL(OnTrackingSelectionChanged(const mitk::DataNode*)));
  this->connect(m_GrabSingleFramePushButton, SIGNAL(pressed()), SIGNAL(OnGrabPressed()));
  this->connect(m_ClearDataPushButton, SIGNAL(pressed()), SIGNAL(OnClearDataPressed()));
  this->connect(m_SaveMatchedDataPushButton, SIGNAL(pressed()), SIGNAL(OnSaveDataPressed()));
  this->connect(m_CalibrationLoadPushButton, SIGNAL(pressed()), SIGNAL(OnLoadCalibrationPressed()));
  this->connect(m_CalibrationRunPushButton, SIGNAL(pressed()), SIGNAL(OnCalibratePressed()));
  this->connect(m_ReconstructVolumePushButton, SIGNAL(pressed()), SIGNAL(OnReconstructPressed()));
  this->SetEnableButtons(true);
}


//-----------------------------------------------------------------------------
USReconGUI::~USReconGUI()
{
}


//-----------------------------------------------------------------------------
void USReconGUI::SetDataStorage(mitk::DataStorage* dataStorage)
{
  mitk::TNodePredicateDataType<mitk::Image>::Pointer isImage = mitk::TNodePredicateDataType<mitk::Image>::New();
  m_ImageComboBox->SetAutoSelectNewItems(false);
  m_ImageComboBox->SetPredicate(isImage);
  m_ImageComboBox->SetDataStorage(dataStorage);
  m_ImageComboBox->setCurrentIndex(0);

  mitk::TNodePredicateDataType<niftk::CoordinateAxesData>::Pointer isMatrix = mitk::TNodePredicateDataType<niftk::CoordinateAxesData>::New();
  m_TrackingComboBox->SetAutoSelectNewItems(false);
  m_TrackingComboBox->SetPredicate(isMatrix);
  m_TrackingComboBox->SetDataStorage(dataStorage);
  m_TrackingComboBox->setCurrentIndex(0);
}


//-----------------------------------------------------------------------------
void USReconGUI::SetEnableButtons(bool isEnabled)
{
  m_GrabSingleFramePushButton->setEnabled(isEnabled);
  m_ClearDataPushButton->setEnabled(isEnabled);
  m_SaveMatchedDataPushButton->setEnabled(isEnabled);
  m_CalibrationLoadPushButton->setEnabled(isEnabled);
  m_CalibrationRunPushButton->setEnabled(isEnabled);
  m_ReconstructVolumePushButton->setEnabled(isEnabled);
}


//-----------------------------------------------------------------------------
void USReconGUI::SetNumberOfFramesLabel(int value)
{
  m_NumberOfFramesValueLabel->setText(QString::number(value));
}


//-----------------------------------------------------------------------------
mitk::DataNode::Pointer USReconGUI::GetImageNode() const
{
  return m_ImageComboBox->GetSelectedNode();
}


//-----------------------------------------------------------------------------
mitk::DataNode::Pointer USReconGUI::GetTrackingNode() const
{
  return m_TrackingComboBox->GetSelectedNode();
}

} // end namespace
