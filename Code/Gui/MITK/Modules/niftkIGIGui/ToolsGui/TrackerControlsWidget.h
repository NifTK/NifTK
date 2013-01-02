/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2012-07-25 07:31:59 +0100 (Wed, 25 Jul 2012) $
 Revision          : $Revision: 9401 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

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
