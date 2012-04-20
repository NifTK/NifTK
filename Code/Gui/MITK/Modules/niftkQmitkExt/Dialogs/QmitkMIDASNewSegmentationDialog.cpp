/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-07-19 12:16:16 +0100 (Tue, 19 Jul 2011) $
 Revision          : $Revision: 6802 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "QmitkMIDASNewSegmentationDialog.h"
#include <QPushButton>
#include <QString>

QmitkMIDASNewSegmentationDialog::QmitkMIDASNewSegmentationDialog(const QColor &defaultColor, QWidget* parent)
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
