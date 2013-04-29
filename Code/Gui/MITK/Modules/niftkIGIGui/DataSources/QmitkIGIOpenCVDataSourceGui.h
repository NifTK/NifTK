/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QMITKIGIULTRASONIXTOOLGUI_H
#define QMITKIGIULTRASONIXTOOLGUI_H

#include "niftkIGIGuiExports.h"
#include <QWidget>
#include <mitkVideoSource.h>
#include <mitkRenderingManager.h>
#include "QmitkIGIDataSourceGui.h"

class QmitkVideoBackground;
class QmitkRenderWindow;
class QmitkIGIOpenCVDataSource;
class QGridLayout;

/**
 * \class QmitkIGIOpenCVDataSourceGui
 * \brief Implements a GUI interface to display live video in the Surgical Guidance plugin.
 */
class NIFTKIGIGUI_EXPORT QmitkIGIOpenCVDataSourceGui : public QmitkIGIDataSourceGui
{
  Q_OBJECT

public:

  mitkClassMacro(QmitkIGIOpenCVDataSourceGui, QmitkIGIDataSourceGui);
  itkNewMacro(QmitkIGIOpenCVDataSourceGui);

protected:

  QmitkIGIOpenCVDataSourceGui(); // Purposefully hidden.
  virtual ~QmitkIGIOpenCVDataSourceGui(); // Purposefully hidden.

  QmitkIGIOpenCVDataSourceGui(const QmitkIGIOpenCVDataSourceGui&); // Purposefully not implemented.
  QmitkIGIOpenCVDataSourceGui& operator=(const QmitkIGIOpenCVDataSourceGui&); // Purposefully not implemented.

  /**
   * \brief Initializes this widget.
   */
  virtual void Initialize(QWidget *parent);

protected slots:

  /**
   * \brief Connected to QmitkIGIOpenCVDataSource::UpdateDisplay to refresh the rendering manager.
   */
  void OnUpdateDisplay();

private:

  QmitkIGIOpenCVDataSource* GetOpenCVDataSource() const;

  QmitkVideoBackground            *m_Background;
  QmitkRenderWindow               *m_RenderWindow;
  mitk::RenderingManager::Pointer  m_RenderingManager;
  QGridLayout                     *m_Layout;

}; // end class

#endif

