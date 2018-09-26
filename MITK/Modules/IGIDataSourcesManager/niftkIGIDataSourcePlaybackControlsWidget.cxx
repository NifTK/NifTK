/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkIGIDataSourcePlaybackControlsWidget.h"
#include <niftkIGIInitialisationDialog.h>
#include <niftkIGIConfigurationDialog.h>
#include <QMessageBox>
#include <QTableWidgetItem>
#include <QVector>
#include <QDateTime>
#include <QTextStream>
#include <QList>
#include <QPainter>

namespace niftk
{

//-----------------------------------------------------------------------------
IGIDataSourcePlaybackControlsWidget::IGIDataSourcePlaybackControlsWidget(
    QWidget *parent)
{
  Ui_IGIDataSourcePlaybackControlsWidget::setupUi(parent);
}


//-----------------------------------------------------------------------------
IGIDataSourcePlaybackControlsWidget::~IGIDataSourcePlaybackControlsWidget()
{
}


} // end namespace
