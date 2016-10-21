/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkDistanceMeasurerGUI.h"
#include <mitkNodePredicateDataType.h>
#include <mitkNodePredicateProperty.h>
#include <mitkNodePredicateAnd.h>
#include <mitkImage.h>

namespace niftk
{

//-----------------------------------------------------------------------------
DistanceMeasurerGUI::DistanceMeasurerGUI(QWidget* parent)
: BaseGUI(parent)
{
  this->setupUi(parent);
  this->connect(m_LeftImageComboBox, SIGNAL(OnSelectionChanged(const mitk::DataNode*)), SIGNAL(LeftImageSelectionChanged(const mitk::DataNode*)));
  this->connect(m_LeftMaskComboBox, SIGNAL(OnSelectionChanged(const mitk::DataNode*)), SIGNAL(LeftMaskSelectionChanged(const mitk::DataNode*)));
  this->connect(m_RightImageComboBox, SIGNAL(OnSelectionChanged(const mitk::DataNode*)), SIGNAL(RightImageSelectionChanged(const mitk::DataNode*)));
  this->connect(m_RightMaskComboBox, SIGNAL(OnSelectionChanged(const mitk::DataNode*)), SIGNAL(RightMaskSelectionChanged(const mitk::DataNode*)));
}


//-----------------------------------------------------------------------------
DistanceMeasurerGUI::~DistanceMeasurerGUI()
{
}


//-----------------------------------------------------------------------------
void DistanceMeasurerGUI::SetDataStorage(mitk::DataStorage* storage)
{
  mitk::TNodePredicateDataType<mitk::Image>::Pointer isImage = mitk::TNodePredicateDataType<mitk::Image>::New();
  mitk::NodePredicateProperty::Pointer isBinary = mitk::NodePredicateProperty::New("binary", mitk::BoolProperty::New(true));
  mitk::NodePredicateProperty::Pointer isNotBinary = mitk::NodePredicateProperty::New("binary", mitk::BoolProperty::New(false));

  mitk::NodePredicateAnd::Pointer isBinaryImage = mitk::NodePredicateAnd::New();
  isBinaryImage->AddPredicate(isImage);
  isBinaryImage->AddPredicate(isBinary);

  mitk::NodePredicateAnd::Pointer isNotBinaryImage = mitk::NodePredicateAnd::New();
  isNotBinaryImage->AddPredicate(isImage);
  isNotBinaryImage->AddPredicate(isNotBinary);

  m_LeftImageComboBox->SetAutoSelectNewItems(false);
  m_LeftImageComboBox->SetPredicate(isNotBinaryImage);
  m_LeftImageComboBox->SetDataStorage(storage);
  m_LeftImageComboBox->setCurrentIndex(0);

  m_LeftMaskComboBox->SetAutoSelectNewItems(false);
  m_LeftMaskComboBox->SetPredicate(isBinaryImage);
  m_LeftMaskComboBox->SetDataStorage(storage);
  m_LeftMaskComboBox->setCurrentIndex(0);

  m_RightImageComboBox->SetAutoSelectNewItems(false);
  m_RightImageComboBox->SetPredicate(isNotBinaryImage);
  m_RightImageComboBox->SetDataStorage(storage);
  m_RightImageComboBox->setCurrentIndex(0);

  m_RightMaskComboBox->SetAutoSelectNewItems(false);
  m_RightMaskComboBox->SetPredicate(isBinaryImage);
  m_RightMaskComboBox->SetDataStorage(storage);
  m_RightMaskComboBox->setCurrentIndex(0);
}


//-----------------------------------------------------------------------------
void DistanceMeasurerGUI::Reset()
{
  m_LeftImageComboBox->setCurrentIndex(0);
  m_LeftMaskComboBox->setCurrentIndex(0);
  m_RightImageComboBox->setCurrentIndex(0);
  m_RightMaskComboBox->setCurrentIndex(0);
}


//-----------------------------------------------------------------------------
void DistanceMeasurerGUI::SetDistance(const double& distance)
{
  m_Distance->setText(QString::number( distance, 'f', 2 ));
}

} // end namespace
