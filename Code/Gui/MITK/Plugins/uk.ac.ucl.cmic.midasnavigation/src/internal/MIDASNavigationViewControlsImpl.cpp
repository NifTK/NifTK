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

MIDASNavigationViewControlsImpl::MIDASNavigationViewControlsImpl(QWidget *parent)
{
  this->setupUi(this);
}

MIDASNavigationViewControlsImpl::~MIDASNavigationViewControlsImpl()
{
}

void MIDASNavigationViewControlsImpl::setupUi(QWidget* parent)
{
  Ui_MIDASNavigationViewControls::setupUi(parent);
}

#endif
