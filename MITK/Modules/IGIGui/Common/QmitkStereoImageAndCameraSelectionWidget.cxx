/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkStereoImageAndCameraSelectionWidget.h"
#include <mitkNodePredicateDataType.h>
#include <mitkNodePredicateProperty.h>
#include <mitkNodePredicateAnd.h>
#include <niftkUndistortion.h>

//-----------------------------------------------------------------------------
QmitkStereoImageAndCameraSelectionWidget::QmitkStereoImageAndCameraSelectionWidget(QWidget *parent)
: m_DataStorage(NULL)
{
  setupUi(this);
  bool ok = false;
  ok = connect(m_LeftImageCombo,  SIGNAL(currentIndexChanged(int)), this, SLOT(OnComboBoxIndexChanged(int)), Qt::QueuedConnection);
  assert(ok);
  ok = connect(m_RightImageCombo, SIGNAL(currentIndexChanged(int)), this, SLOT(OnComboBoxIndexChanged(int)), Qt::QueuedConnection);
  assert(ok);
}


//-----------------------------------------------------------------------------
QmitkStereoImageAndCameraSelectionWidget::~QmitkStereoImageAndCameraSelectionWidget()
{
  bool ok = false;
  ok = disconnect(m_LeftImageCombo,  SIGNAL(currentIndexChanged(int)), this, SLOT(OnComboBoxIndexChanged(int)));
  assert(ok);
  ok = disconnect(m_RightImageCombo, SIGNAL(currentIndexChanged(int)), this, SLOT(OnComboBoxIndexChanged(int)));
  assert(ok);
}


//-----------------------------------------------------------------------------
void QmitkStereoImageAndCameraSelectionWidget::SetRightChannelEnabled(const bool& isEnabled)
{
  m_RightImageCombo->setEnabled(isEnabled);
  m_RightMaskCombo->setEnabled(isEnabled);
}


//-----------------------------------------------------------------------------
void QmitkStereoImageAndCameraSelectionWidget::SetLeftChannelEnabled(const bool& isEnabled)
{
  m_LeftImageCombo->setEnabled(isEnabled);
  m_LeftMaskCombo->setEnabled(isEnabled);
}


//-----------------------------------------------------------------------------
mitk::Image* QmitkStereoImageAndCameraSelectionWidget::GetLeftImage() const
{
  mitk::Image* result = NULL;

  mitk::DataNode* node = this->GetLeftNode();
  if (node != NULL)
  {
    result = dynamic_cast<mitk::Image*>(node->GetData());
  }

  return result;
}


//-----------------------------------------------------------------------------
mitk::DataNode* QmitkStereoImageAndCameraSelectionWidget::GetLeftNode() const
{
  return m_LeftImageCombo->GetSelectedNode();
}


//-----------------------------------------------------------------------------
mitk::Image* QmitkStereoImageAndCameraSelectionWidget::GetRightImage() const
{
  mitk::Image* result = NULL;

  mitk::DataNode* node = this->GetRightNode();
  if (node != NULL)
  {
    result = dynamic_cast<mitk::Image*>(node->GetData());
  }

  return result;
}


//-----------------------------------------------------------------------------
mitk::DataNode* QmitkStereoImageAndCameraSelectionWidget::GetRightNode() const
{
  return m_RightImageCombo->GetSelectedNode();
}


//-----------------------------------------------------------------------------
mitk::Image* QmitkStereoImageAndCameraSelectionWidget::GetLeftMask() const
{
  mitk::Image* result = NULL;

  mitk::DataNode* node = this->GetLeftMaskNode();
  if (node != NULL)
  {
    result = dynamic_cast<mitk::Image*>(node->GetData());
  }

  return result;
}


//-----------------------------------------------------------------------------
mitk::DataNode* QmitkStereoImageAndCameraSelectionWidget::GetLeftMaskNode() const
{
  return m_LeftMaskCombo->GetSelectedNode();
}


//-----------------------------------------------------------------------------
mitk::Image* QmitkStereoImageAndCameraSelectionWidget::GetRightMask() const
{
  mitk::Image* result = NULL;

  mitk::DataNode* node = this->GetRightMaskNode();
  if (node != NULL)
  {
    result = dynamic_cast<mitk::Image*>(node->GetData());
  }

  return result;
}


//-----------------------------------------------------------------------------
mitk::DataNode* QmitkStereoImageAndCameraSelectionWidget::GetRightMaskNode() const
{
  return m_RightMaskCombo->GetSelectedNode();
}


//-----------------------------------------------------------------------------
mitk::DataNode* QmitkStereoImageAndCameraSelectionWidget::GetCameraNode() const
{
  return m_CameraPositionComboBox->GetSelectedNode();
}


//-----------------------------------------------------------------------------
void QmitkStereoImageAndCameraSelectionWidget::SetCameraNode(mitk::DataNode* node)
{
  m_CameraPositionComboBox->SetSelectedNode(node);
}


//-----------------------------------------------------------------------------
niftk::CoordinateAxesData* QmitkStereoImageAndCameraSelectionWidget::GetCameraTransform() const
{
  niftk::CoordinateAxesData* result = NULL;

  mitk::DataNode* node = this->GetCameraNode();
  if (node != NULL)
  {
    result = dynamic_cast<niftk::CoordinateAxesData*>(node->GetData());
  }

  return result;
}


//-----------------------------------------------------------------------------
void QmitkStereoImageAndCameraSelectionWidget::SetDataStorage(const mitk::DataStorage* dataStorage)
{
  m_DataStorage = const_cast<mitk::DataStorage*>(dataStorage);

  mitk::TNodePredicateDataType<mitk::Image>::Pointer isImage = mitk::TNodePredicateDataType<mitk::Image>::New();
  mitk::NodePredicateProperty::Pointer isBinary = mitk::NodePredicateProperty::New("binary", mitk::BoolProperty::New(true));
  mitk::NodePredicateProperty::Pointer isNotBinary = mitk::NodePredicateProperty::New("binary", mitk::BoolProperty::New(false));

  mitk::NodePredicateAnd::Pointer isBinaryImage = mitk::NodePredicateAnd::New();
  isBinaryImage->AddPredicate(isImage);
  isBinaryImage->AddPredicate(isBinary);

  mitk::NodePredicateAnd::Pointer isNonBinaryImage = mitk::NodePredicateAnd::New();
  isNonBinaryImage->AddPredicate(isImage);
  isNonBinaryImage->AddPredicate(isNotBinary);

  m_LeftImageCombo->SetAutoSelectNewItems(false);
  m_LeftImageCombo->SetPredicate(isNonBinaryImage);
  m_LeftImageCombo->SetDataStorage(m_DataStorage);

  m_RightImageCombo->SetAutoSelectNewItems(false);
  m_RightImageCombo->SetPredicate(isNonBinaryImage);
  m_RightImageCombo->SetDataStorage(m_DataStorage);

  m_LeftMaskCombo->SetAutoSelectNewItems(false);
  m_LeftMaskCombo->SetPredicate(isBinaryImage);
  m_LeftMaskCombo->SetDataStorage(m_DataStorage);

  m_RightMaskCombo->SetAutoSelectNewItems(false);
  m_RightMaskCombo->SetPredicate(isBinaryImage);
  m_RightMaskCombo->SetDataStorage(m_DataStorage);

  mitk::TNodePredicateDataType<niftk::CoordinateAxesData>::Pointer isCoords = mitk::TNodePredicateDataType<niftk::CoordinateAxesData>::New();
  m_CameraPositionComboBox->SetDataStorage(m_DataStorage);
  m_CameraPositionComboBox->SetAutoSelectNewItems(false);
  m_CameraPositionComboBox->SetPredicate(isCoords);
}


//-----------------------------------------------------------------------------
void QmitkStereoImageAndCameraSelectionWidget::OnComboBoxIndexChanged(int index)
{
  UpdateNodeNameComboBox();
}


//-----------------------------------------------------------------------------
template <typename T>
static bool HasCalibProp(const typename T::Pointer& n)
{
  mitk::BaseProperty::Pointer  bp = n->GetProperty(niftk::Undistortion::s_CameraCalibrationPropertyName);
  if (bp.IsNull())
  {
    return false;
  }
  return true;
}


//-----------------------------------------------------------------------------
void QmitkStereoImageAndCameraSelectionWidget::UpdateNodeNameComboBox()
{
  if(m_DataStorage.IsNotNull())
  {
    QString leftText  = m_LeftImageCombo->currentText();
    QString rightText = m_RightImageCombo->currentText();

    mitk::DataNode::Pointer   leftNode  = m_DataStorage->GetNamedNode(leftText.toStdString());
    mitk::DataNode::Pointer   rightNode = m_DataStorage->GetNamedNode(rightText.toStdString());

    // either node or attached image has to have calibration property
    if (leftNode.IsNotNull())
    {
      bool    leftHasProp = HasCalibProp<mitk::DataNode>(leftNode);
      if (!leftHasProp)
      {
        // note: our comboboxes should have nodes only with image data!
        mitk::Image::Pointer img = dynamic_cast<mitk::Image*>(leftNode->GetData());
        assert(img.IsNotNull());
        leftHasProp = HasCalibProp<mitk::Image>(img);
      }

      if (leftHasProp)
      {
        m_LeftImageCombo->setStyleSheet("background-color: rgb(200, 255, 200);");
      }
      else
      {
        m_LeftImageCombo->setStyleSheet("background-color: rgb(255, 200, 200);");
      }
    }

    if (rightNode.IsNotNull())
    {
      bool    rightHasProp = HasCalibProp<mitk::DataNode>(rightNode);
      if (!rightNode)
      {
        // note: our comboboxes should have nodes only with image data!
        mitk::Image::Pointer img = dynamic_cast<mitk::Image*>(rightNode->GetData());
        assert(img.IsNotNull());
        rightHasProp = HasCalibProp<mitk::Image>(img);
      }

      if (rightHasProp)
      {
        m_RightImageCombo->setStyleSheet("background-color: rgb(200, 255, 200);");
      }
      else
      {
        m_RightImageCombo->setStyleSheet("background-color: rgb(255, 200, 200);");
      }
    }
  }
}
