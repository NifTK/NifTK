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
    m_ReferenceImageNameLabel->setText(QString("<font color='black'>%1</font>").arg(referenceImageName));
    m_ReferenceImageNameLabel->setToolTip(referenceImageName);
  }
  else
  {
    m_ReferenceImageNameLabel->setText("<font color='red'>&lt;not selected&gt;</font>");
    m_ReferenceImageNameLabel->setToolTip("&lt;not selected&gt;");
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
  }
  else
  {
    m_SegmentationImageNameLabel->setText("<font color='red'>&lt;not selected&gt;</font>");
    m_SegmentationImageNameLabel->setToolTip("&lt;not selected&gt;");
  }

  m_NewSegmentationButton->setEnabled(hasReferenceData && !hasWorkingData);
}

}
