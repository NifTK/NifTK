/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkIGIDataSourcePlaybackControlsWidget_h
#define niftkIGIDataSourcePlaybackControlsWidget_h

#include "niftkIGIDataSourcesManagerExports.h"
#include "ui_niftkIGIDataSourcePlaybackControlsWidget.h"
#include "niftkIGIDataSourceManager.h"

#include <mitkDataStorage.h>
#include <QWidget>
#include <QTimer>
#include <QTime>

namespace niftk
{

/**
 * \class IGIDataSourcePlaybackWidget
 * \brief Widget class to manage play back of a group of IGIDataSources (trackers, ultra-sound machines, video etc).
 *
 * This class must delegate all functionality to IGIDataSourceManager, and should
 * only contain Widget related stuff. Conversely, IGIDataSourceManager should
 * only contain non-Widget related stuff.
 */
class NIFTKIGIDATASOURCESMANAGER_EXPORT IGIDataSourcePlaybackControlsWidget :
    public QWidget,
    public Ui_IGIDataSourcePlaybackControlsWidget
{

  Q_OBJECT

public:

  IGIDataSourcePlaybackControlsWidget(QWidget *parent = 0);

  virtual ~IGIDataSourcePlaybackControlsWidget();

signals:

protected:

  IGIDataSourcePlaybackControlsWidget(const IGIDataSourcePlaybackControlsWidget&); // Purposefully not implemented.
  IGIDataSourcePlaybackControlsWidget& operator=(const IGIDataSourcePlaybackControlsWidget&); // Purposefully not implemented.

private slots:

private:

}; // end class;

} // end namespace

#endif
