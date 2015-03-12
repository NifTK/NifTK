/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkVideoTestClient.h"
#include <QApplication>
#include <QMessageBox>


//-----------------------------------------------------------------------------
QmitkVideoTestClient::QmitkVideoTestClient(
    const std::string& hostname,
    const int& portNumber,
    const int& numberOfSeconds,
    QWidget *parent)
: QWidget(parent)
{
}


//-----------------------------------------------------------------------------
QmitkVideoTestClient::~QmitkVideoTestClient()
{
}


//-----------------------------------------------------------------------------
void QmitkVideoTestClient::Run()
{

}
