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
    m_ToolManager->ReferenceDataChanged -= mitk::MessageDelegate<SegmentationSelectorWidget>(this, &SegmentationSelectorWidget::UpdateWidgets);
    m_ToolManager->WorkingDataChanged -= mitk::MessageDelegate<SegmentationSelectorWidget>(this, &SegmentationSelectorWidget::UpdateWidgets);
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
      m_ToolManager->ReferenceDataChanged -= mitk::MessageDelegate<SegmentationSelectorWidget>(this, &SegmentationSelectorWidget::UpdateWidgets);
      m_ToolManager->WorkingDataChanged -= mitk::MessageDelegate<SegmentationSelectorWidget>(this, &SegmentationSelectorWidget::UpdateWidgets);
    }

    if (toolManager)
    {
      toolManager->ReferenceDataChanged += mitk::MessageDelegate<SegmentationSelectorWidget>(this, &SegmentationSelectorWidget::UpdateWidgets);
      toolManager->WorkingDataChanged += mitk::MessageDelegate<SegmentationSelectorWidget>(this, &SegmentationSelectorWidget::UpdateWidgets);
    }

    m_ToolManager = toolManager;

    this->UpdateWidgets();
  }
}


//-----------------------------------------------------------------------------
void SegmentationSelectorWidget::UpdateWidgets()
{
  bool hasReferenceNode = m_ToolManager && !m_ToolManager->GetReferenceData().empty();
  bool hasWorkingNode = m_ToolManager && !m_ToolManager->GetWorkingData().empty();

  if (!hasReferenceNode && !hasWorkingNode)
  {
    QString toolTip = "Choose a reference or segmentation image from the Data Manager\n"
                      "that has the same geometry as the selected viewer.";

    m_ReferenceNodeNameLabel->setText("<font color='red'>&lt;not selected&gt;</font>");
    m_ReferenceNodeNameLabel->setToolTip(toolTip);

    m_SegmentationNodeNameLabel->setText("<font color='red'>&lt;not selected&gt;</font>");
    m_SegmentationNodeNameLabel->setToolTip(toolTip);

    m_NewSegmentationButton->setEnabled(false);
    m_NewSegmentationButton->setToolTip(toolTip);
  }
  else if (hasReferenceNode && !hasWorkingNode)
  {
    QString referenceImageName = QString::fromStdString(m_ToolManager->GetReferenceData(0)->GetName());
    m_ReferenceNodeNameLabel->setText(QString("<font color='black'>%1</font>").arg(referenceImageName));
    m_ReferenceNodeNameLabel->setToolTip(referenceImageName);

    m_SegmentationNodeNameLabel->setText("<font color='red'>&lt;not selected&gt;</font>");
    m_SegmentationNodeNameLabel->setToolTip("Hit the 'Start/restart segmentation' button to create\n"
                                            "a new segmentation, or select a segmentation image\n"
                                            "whose parent is a reference image with the same\n"
                                            "geometry as the selected viewer.");

    m_NewSegmentationButton->setEnabled(true);
    m_NewSegmentationButton->setToolTip("Creates a new segmentation.");
  }
  else if (!hasReferenceNode && hasWorkingNode)
  {
    /// This should not happen, really.
    m_ReferenceNodeNameLabel->setText("<font color='red'>&lt;not selected&gt;</font>");
    m_ReferenceNodeNameLabel->setToolTip("");

    m_SegmentationNodeNameLabel->setText("<font color='red'>&lt;not selected&gt;</font>");
    m_SegmentationNodeNameLabel->setToolTip("");

    m_NewSegmentationButton->setEnabled(false);
    m_NewSegmentationButton->setToolTip("");
  }
  else
  {
    QString referenceImageName = QString::fromStdString(m_ToolManager->GetReferenceData(0)->GetName());
    m_ReferenceNodeNameLabel->setText(QString("<font color='black'>%1</font>").arg(referenceImageName));
    m_ReferenceNodeNameLabel->setToolTip(referenceImageName);

    QString segmentationNodeName = QString::fromStdString(m_ToolManager->GetWorkingData(0)->GetName());
    m_SegmentationNodeNameLabel->setText(QString("<font color='black'>%1</font>").arg(segmentationNodeName));
    m_SegmentationNodeNameLabel->setToolTip(segmentationNodeName);

    m_NewSegmentationButton->setEnabled(false);
    m_NewSegmentationButton->setToolTip("A segmentation is already in progress. Choose a different\n"
                                        "image or finalise (OK) or discard (Cancel) the current\n"
                                        "segmentation to start a new one.");
  }
}

}
