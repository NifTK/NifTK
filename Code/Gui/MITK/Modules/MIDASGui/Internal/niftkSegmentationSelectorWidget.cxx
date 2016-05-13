/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkSegmentationSelectorWidget.h"

namespace niftk
{

//-----------------------------------------------------------------------------
SegmentationSelectorWidget::SegmentationSelectorWidget(QWidget *parent)
: QWidget(parent)
{
  this->setupUi(parent);

  this->SelectReferenceImage();
  this->SelectSegmentationImage();

  this->connect(m_NewSegmentationButton, SIGNAL(clicked()), SIGNAL(NewSegmentationButtonClicked()));
}


//-----------------------------------------------------------------------------
SegmentationSelectorWidget::~SegmentationSelectorWidget()
{
}


//-----------------------------------------------------------------------------
void SegmentationSelectorWidget::SelectReferenceImage(const QString& imageName)
{
  QString labelText = imageName.isNull()
      ? "<font color='red'>&lt;not selected&gt;</font>"
      : QString("<font color='black'>%1</font>").arg(imageName);

  m_ReferenceImageNameLabel->setText(labelText);
}


//-----------------------------------------------------------------------------
void SegmentationSelectorWidget::SelectSegmentationImage(const QString& imageName)
{
  QString labelText = imageName.isNull()
      ? "<font color='red'>&lt;not selected&gt;</font>"
      : QString("<font color='black'>%1</font>").arg(imageName);

  bool referenceImageSelected = m_ReferenceImageNameLabel->text() != QString("<font color='red'>&lt;not selected&gt;</font>");
  m_NewSegmentationButton->setEnabled(imageName.isNull() && referenceImageSelected);

  m_SegmentationImageNameLabel->setText(labelText);
}

}
