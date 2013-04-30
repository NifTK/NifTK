/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkMIDASNewSegmentationDialog.h"

#include <QPushButton>
#include <QString>

//-----------------------------------------------------------------------------
QmitkMIDASNewSegmentationDialog::QmitkMIDASNewSegmentationDialog(const QColor& defaultColor, QWidget* parent)
: QmitkNewSegmentationDialog(parent)
{
  QString styleSheet = "background-color: rgb(";
  styleSheet.append(QString::number(defaultColor.red()));
  styleSheet.append(",");
  styleSheet.append(QString::number(defaultColor.green()));
  styleSheet.append(",");
  styleSheet.append(QString::number(defaultColor.blue()));
  styleSheet.append(")");

  btnColor->setStyleSheet(styleSheet);

  m_Color.setRedF(defaultColor.redF());
  m_Color.setGreenF(defaultColor.greenF());
  m_Color.setBlueF(defaultColor.blueF());
}


//-----------------------------------------------------------------------------
QmitkMIDASNewSegmentationDialog::~QmitkMIDASNewSegmentationDialog()
{
}
