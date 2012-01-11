/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-12-05 18:07:46 +0000 (Mon, 05 Dec 2011) $
 Revision          : $Revision: 7922 $
 Last modified by  : $Author: mjc $

 Original author   : a.duttaroy@cs.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef MIDASNAVIGATIONVIEWCONTROLSIMPL_CPP
#define MIDASNAVIGATIONVIEWCONTROLSIMPL_CPP

#include "MIDASNavigationViewControlsImpl.h"
#include <iostream>
#include <QGridLayout>
#include <QVBoxLayout>

MIDASNavigationViewControlsImpl::MIDASNavigationViewControlsImpl()
{
}

MIDASNavigationViewControlsImpl::~MIDASNavigationViewControlsImpl()
{

}

void MIDASNavigationViewControlsImpl::setupUi(QWidget* parent)
{
  Ui_MIDASNavigationViewControls::setupUi(parent);

  m_AxialRadioButton->setChecked(true);
  m_SagittalRadioButton->setChecked(false);
  m_CoronalRadioButton->setChecked(false);
  m_MagnificationFactorWidget->setToolTip("changes the magnification of the currently selected view (red outline).");
  m_SliceSelectionWidget->setToolTip("changes the slice number of the currently selected view (red outline), where slice numbering starts at one.");
  m_SliceSelectionWidget->SetOffset(1);
  m_TimeSelectionWidget->SetText("time");
  m_TimeSelectionWidget->setToolTip("changes the time step number of the currently selected view (red outline), where time number starts at zero.");
}

#endif
