/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "TrackerControlsWidget.h"

#include <iostream>
#include <exception>
#include <cmath>
#include <QFile>
#include <QPixmap>

//-----------------------------------------------------------------------------
TrackerControlsWidget::TrackerControlsWidget(QWidget *parent)
  : QWidget()
{
  setupUi(this);

  m_CurrentlyTrackedTools.clear();

  // Extract the ROM files from qrc
  /*
  QFile::copy(":/NiftyLink/8700338.rom", "8700338.rom");
  QFile::copy(":/NiftyLink/8700339.rom", "8700339.rom");
  QFile::copy(":/NiftyLink/8700340.rom", "8700340.rom");
  QFile::copy(":/NiftyLink/8700302.rom", "8700302.rom");
  */

  // Temporary:
  pushButton_Tracking->setVisible(false);
  toolButton_Assoc->setVisible(true);
}


//-----------------------------------------------------------------------------
TrackerControlsWidget::~TrackerControlsWidget(void)
{
}


//-----------------------------------------------------------------------------
void TrackerControlsWidget::InitTrackerTools(QStringList &toolList)
{
  QPixmap pix(22, 22);
  pix.fill(QColor(Qt::lightGray));

  QPixmap pix2(22, 22);
  pix2.fill(QColor("green"));

  comboBox_trackerTool->clear();

  QString thistool;
  foreach (thistool, toolList)
  {
	  comboBox_trackerTool->addItem(pix2, thistool);
  }

  m_CurrentlyTrackedTools = toolList;
}


//-----------------------------------------------------------------------------
void TrackerControlsWidget::ToggleTrackerTool(QString &outputName, bool &outputEnabled)
{
  outputName = comboBox_trackerTool->currentText();
  outputEnabled = false;

  // Tracking of tool is currently enabled, so disable it.
  if (m_CurrentlyTrackedTools.contains(outputName))
  {
    int i = m_CurrentlyTrackedTools.indexOf(outputName);
    m_CurrentlyTrackedTools.removeAt(i);

    QPixmap pix(22, 22);
    pix.fill(QColor(Qt::lightGray));

    int index = comboBox_trackerTool->currentIndex();
    comboBox_trackerTool->removeItem(index);
    comboBox_trackerTool->insertItem(index, pix, outputName);
    comboBox_trackerTool->setCurrentIndex(index);

    outputEnabled = false;
  }
  else // Tracking of tool currently disabled, so enable it.
  {
    m_CurrentlyTrackedTools.append(outputName);

    QPixmap pix(22, 22);
    pix.fill(QColor("green"));

    int index = comboBox_trackerTool->currentIndex();
    comboBox_trackerTool->removeItem(index);
    comboBox_trackerTool->insertItem(index, pix, outputName);
    comboBox_trackerTool->setCurrentIndex(index);

    outputEnabled = true;
  }
}


//-----------------------------------------------------------------------------
QString TrackerControlsWidget::GetCurrentToolName() const
{
  return comboBox_trackerTool->currentText();
}
