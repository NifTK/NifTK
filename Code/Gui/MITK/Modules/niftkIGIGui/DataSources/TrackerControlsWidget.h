/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef TRACKERCONTROLSWIDGET_H
#define TRACKERCONTROLSWIDGET_H

#include <QWidget>
#include "niftkIGIGuiExports.h"
#include "ui_TrackerControlsWidget.h"

/**
 * \class TrackerControlsWidget
 * \brief Implements some basic functionality for stopping/starting a generic tracker.
 */

class NIFTKIGIGUI_EXPORT TrackerControlsWidget : public QWidget, public Ui_TrackerControlsWidget
{
  Q_OBJECT

public:

  /// \brief Basic constructor.
  TrackerControlsWidget(QWidget *parent = 0);

  /// \brief Basic destructor.
  ~TrackerControlsWidget(void);

  /// \brief Sets up the combo box with a list of tracker tool rom files.
  void InitTrackerTools(QStringList &toolList);

  /// \brief Asks the widget to toggle the current tool, and this method
  /// returns the name of the tool toggled, and the desired status.
  void ToggleTrackerTool(QString &outputName, bool &outputEnabled);

  /// \brief Returns the name of the current tool in the combo box.
  QString GetCurrentToolName() const;

signals:

protected:

private slots:

private:

  QStringList m_CurrentlyTrackedTools;

};

#endif
