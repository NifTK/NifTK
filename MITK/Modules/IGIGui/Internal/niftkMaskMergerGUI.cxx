/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkMaskMergerGUI.h"
#include <mitkNodePredicateDataType.h>
#include <mitkNodePredicateProperty.h>
#include <mitkNodePredicateAnd.h>
#include <mitkImage.h>

namespace niftk
{

//-----------------------------------------------------------------------------
MaskMergerGUI::MaskMergerGUI(QWidget* parent)
: BaseGUI(parent)
{
  this->setupUi(parent);
  this->connect(m_LeftMask1ComboBox, SIGNAL(OnSelectionChanged(const mitk::DataNode*)), SIGNAL(LeftMask1SelectionChanged(const mitk::DataNode*)));
  this->connect(m_LeftMask2ComboBox, SIGNAL(OnSelectionChanged(const mitk::DataNode*)), SIGNAL(LeftMask2SelectionChanged(const mitk::DataNode*)));
  this->connect(m_RightMask1ComboBox, SIGNAL(OnSelectionChanged(const mitk::DataNode*)), SIGNAL(RightMask1SelectionChanged(const mitk::DataNode*)));
  this->connect(m_RightMask2ComboBox, SIGNAL(OnSelectionChanged(const mitk::DataNode*)), SIGNAL(RightMask2SelectionChanged(const mitk::DataNode*)));
}


//-----------------------------------------------------------------------------
MaskMergerGUI::~MaskMergerGUI()
{
}


//-----------------------------------------------------------------------------
void MaskMergerGUI::ResetLeft()
{
  m_LeftMask1ComboBox->setCurrentIndex(0);
  m_LeftMask2ComboBox->setCurrentIndex(0);
}


//-----------------------------------------------------------------------------
void MaskMergerGUI::ResetRight()
{
  m_RightMask1ComboBox->setCurrentIndex(0);
  m_RightMask2ComboBox->setCurrentIndex(0);
}


//-----------------------------------------------------------------------------
void MaskMergerGUI::SetDataStorage(mitk::DataStorage* storage)
{
  mitk::TNodePredicateDataType<mitk::Image>::Pointer isImage = mitk::TNodePredicateDataType<mitk::Image>::New();
  mitk::NodePredicateProperty::Pointer isBinary = mitk::NodePredicateProperty::New("binary", mitk::BoolProperty::New(true));

  mitk::NodePredicateAnd::Pointer isBinaryImage = mitk::NodePredicateAnd::New();
  isBinaryImage->AddPredicate(isImage);
  isBinaryImage->AddPredicate(isBinary);

  m_LeftMask1ComboBox->SetAutoSelectNewItems(false);
  m_LeftMask1ComboBox->SetPredicate(isBinaryImage);
  m_LeftMask1ComboBox->SetDataStorage(storage);
  m_LeftMask1ComboBox->setCurrentIndex(0);

  m_LeftMask2ComboBox->SetAutoSelectNewItems(false);
  m_LeftMask2ComboBox->SetPredicate(isBinaryImage);
  m_LeftMask2ComboBox->SetDataStorage(storage);
  m_LeftMask2ComboBox->setCurrentIndex(0);

  m_RightMask1ComboBox->SetAutoSelectNewItems(false);
  m_RightMask1ComboBox->SetPredicate(isBinaryImage);
  m_RightMask1ComboBox->SetDataStorage(storage);
  m_RightMask1ComboBox->setCurrentIndex(0);

  m_RightMask2ComboBox->SetAutoSelectNewItems(false);
  m_RightMask2ComboBox->SetPredicate(isBinaryImage);
  m_RightMask2ComboBox->SetDataStorage(storage);
  m_RightMask2ComboBox->setCurrentIndex(0);
}

} // end namespace
