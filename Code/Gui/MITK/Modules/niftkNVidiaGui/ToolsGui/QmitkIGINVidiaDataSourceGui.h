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

#include "niftkNVidiaGuiExports.h"
#include <QWidget>
#include "QmitkIGIDataSourceGui.h"
#include "ui_QmitkIGINVidiaDataSourceGui.h"

class QmitkIGINVidiaDataSource;
class QLabel;
class QGridLayout;

/**
 * \class QmitkIGIOpenCVDataSourceGui
 * \brief Implements a GUI interface to display live video from NVidia SDI in the Surgical Guidance plugin.
 */
class NIFTKNVIDIAGUI_EXPORT QmitkIGINVidiaDataSourceGui : public QmitkIGIDataSourceGui, public Ui_QmitkIGINVidiaDataSourceGui
{
  Q_OBJECT

public:

  mitkClassMacro(QmitkIGINVidiaDataSourceGui, QmitkIGIDataSourceGui);
  itkNewMacro(QmitkIGINVidiaDataSourceGui);

protected:

  QmitkIGINVidiaDataSourceGui(); // Purposefully hidden.
  virtual ~QmitkIGINVidiaDataSourceGui(); // Purposefully hidden.

  QmitkIGINVidiaDataSourceGui(const QmitkIGINVidiaDataSourceGui&); // Purposefully not implemented.
  QmitkIGINVidiaDataSourceGui& operator=(const QmitkIGINVidiaDataSourceGui&); // Purposefully not implemented.

  /**
   * \brief Initializes this widget.
   */
  virtual void Initialize(QWidget *parent);

protected slots:

  /**
   * \brief Connected to QmitkIGINVidiaDataSource::UpdateDisplay to refresh the rendering manager.
   */
  void OnUpdateDisplay();

private:

  QmitkIGINVidiaDataSource* GetQmitkIGINVidiaDataSource() const;

  QLabel      *m_DisplayWidget;
  QGridLayout *m_Layout;

}; // end class

#endif

