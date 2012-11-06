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

  if (toolList.contains(QString("8700338.rom")))
    comboBox_trackerTool->addItem(pix2, "8700338.rom");
  else
    comboBox_trackerTool->addItem(pix, "8700338.rom");

  if (toolList.contains(QString("8700339.rom")))
    comboBox_trackerTool->addItem(pix2, "8700339.rom");
  else
    comboBox_trackerTool->addItem(pix, "8700339.rom");

  if (toolList.contains(QString("8700340.rom")))
    comboBox_trackerTool->addItem(pix2, "8700340.rom");
  else
    comboBox_trackerTool->addItem(pix, "8700340.rom");

  if (toolList.contains(QString("8700302.rom")))
    comboBox_trackerTool->addItem(pix2, "8700302.rom");
  else
    comboBox_trackerTool->addItem(pix, "8700302.rom");

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
