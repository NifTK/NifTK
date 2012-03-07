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

QmitkMIDASNewSegmentationDialog::QmitkMIDASNewSegmentationDialog(QWidget* parent)
: QmitkNewSegmentationDialog(parent)
{
  btnColor->setStyleSheet("background-color:rgb(0,255,0)");
  m_Color.setRedF(0);
  m_Color.setGreenF(1);
  m_Color.setBlueF(0);
}
