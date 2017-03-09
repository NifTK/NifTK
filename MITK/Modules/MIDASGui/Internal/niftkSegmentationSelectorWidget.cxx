/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkSegmentationSelectorWidget.h"

#include <mitkToolManager.h>

namespace niftk
{

//-----------------------------------------------------------------------------
SegmentationSelectorWidget::SegmentationSelectorWidget(QWidget* parent)
: QWidget(parent),
  m_ToolManager(nullptr)
{
  this->setupUi(parent);

  this->connect(m_NewSegmentationButton, SIGNAL(clicked()), SIGNAL(NewSegmentationButtonClicked()));
}


//-----------------------------------------------------------------------------
SegmentationSelectorWidget::~SegmentationSelectorWidget()
{
  if (m_ToolManager)
  {
    m_ToolManager->ReferenceDataChanged -= mitk::MessageDelegate<SegmentationSelectorWidget>(this, &SegmentationSelectorWidget::OnReferenceDataChanged);
    m_ToolManager->WorkingDataChanged -= mitk::MessageDelegate<SegmentationSelectorWidget>(this, &SegmentationSelectorWidget::OnWorkingDataChanged);
  }
}


//-----------------------------------------------------------------------------
mitk::ToolManager* SegmentationSelectorWidget::GetToolManager() const
{
  return m_ToolManager;
}


//-----------------------------------------------------------------------------
void SegmentationSelectorWidget::SetToolManager(mitk::ToolManager* toolManager)
{
  if (toolManager != m_ToolManager)
  {
    if (m_ToolManager)
    {
      m_ToolManager->ReferenceDataChanged -= mitk::MessageDelegate<SegmentationSelectorWidget>(this, &SegmentationSelectorWidget::OnReferenceDataChanged);
      m_ToolManager->WorkingDataChanged -= mitk::MessageDelegate<SegmentationSelectorWidget>(this, &SegmentationSelectorWidget::OnWorkingDataChanged);
    }

    if (toolManager)
    {
      toolManager->ReferenceDataChanged += mitk::MessageDelegate<SegmentationSelectorWidget>(this, &SegmentationSelectorWidget::OnReferenceDataChanged);
      toolManager->WorkingDataChanged += mitk::MessageDelegate<SegmentationSelectorWidget>(this, &SegmentationSelectorWidget::OnWorkingDataChanged);
    }

    m_ToolManager = toolManager;

    this->OnReferenceDataChanged();
    this->OnWorkingDataChanged();
  }
}


//-----------------------------------------------------------------------------
void SegmentationSelectorWidget::OnReferenceDataChanged()
{
  bool hasReferenceData = m_ToolManager && !m_ToolManager->GetReferenceData().empty();
  bool hasWorkingData = m_ToolManager && !m_ToolManager->GetWorkingData().empty();

  if (hasReferenceData)
  {
    QString referenceImageName = QString::fromStdString(m_ToolManager->GetReferenceData(0)->GetName());
    m_ReferenceImageNameLabel->setText(referenceImageName);
    m_ReferenceImageNameLabel->setToolTip(referenceImageName);
  }
  else
  {
    m_ReferenceImageNameLabel->setText("<font color='red'>&lt;not selected&gt;</font>");
    m_ReferenceImageNameLabel->setToolTip("Choose a reference image from the Data Manager.");
  }

  if (!hasReferenceData)
  {
    m_NewSegmentationButton->setToolTip(
          "Choose a reference image from the Data Manager.");
  }
  else if (!hasWorkingData)
  {
    m_NewSegmentationButton->setToolTip(
          "Hit the 'Start/restart segmentation' button after selecting\n"
          "a reference image or a segmentation image in the Data Manager.");
  }
  else
  {
    m_NewSegmentationButton->setToolTip(
          "Only one segmentation can be edited at a time. You need to finalise (OK)\n"
          "or discard (Cancel) the current segmentation to start a new one.");
  }
  m_NewSegmentationButton->setEnabled(hasReferenceData && !hasWorkingData);
}


//-----------------------------------------------------------------------------
void SegmentationSelectorWidget::OnWorkingDataChanged()
{
  bool hasReferenceData = m_ToolManager && !m_ToolManager->GetReferenceData().empty();
  bool hasWorkingData = m_ToolManager && !m_ToolManager->GetWorkingData().empty();

  if (hasWorkingData)
  {
    QString segmentationImageName = QString::fromStdString(m_ToolManager->GetWorkingData(0)->GetName());
    m_SegmentationImageNameLabel->setText(QString("<font color='black'>%1</font>").arg(segmentationImageName));
    m_SegmentationImageNameLabel->setToolTip(QString("<font color='black'>%1</font>").arg(segmentationImageName));
    m_SegmentationImageNameLabel->setToolTip(
          QString("The segmentation '%1' is being edited. Only one segmentation can be\n"
                  "edited at a time. You need to finalise (OK) or discard (Cancel) the\n"
                  "current segmentation to start a new one.").arg(segmentationImageName));
  }
  else
  {
    m_SegmentationImageNameLabel->setText("<font color='red'>&lt;not selected&gt;</font>");
    m_SegmentationImageNameLabel->setToolTip(
          "There is no segmentation in progress. Choose a reference image\n"
          "or a segmentation from the Data Manager and hit the 'Start/restart\n"
          "segmentation' button.");
  }

  if (!hasReferenceData)
  {
    m_NewSegmentationButton->setToolTip(
          "Choose a reference image from the Data Manager.");
  }
  else if (!hasWorkingData)
  {
    m_NewSegmentationButton->setToolTip(
          "Hit the 'Start/restart segmentation' button after selecting\n"
          "a reference image or a segmentation image in the Data Manager.");
  }
  else
  {
    m_NewSegmentationButton->setToolTip(
          "Only one segmentation can be edited at a time. You need to finalise (OK)\n"
          "or discard (Cancel) the current segmentation to start a new one.");
  }
  m_NewSegmentationButton->setEnabled(hasReferenceData && !hasWorkingData);
}

}
