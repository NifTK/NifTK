/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkIGINVidiaDataSourceGui_h
#define QmitkIGINVidiaDataSourceGui_h

#include "niftkNVidiaGuiExports.h"
#include <QWidget>
#include <QmitkIGIDataSourceGui.h>
#include "ui_QmitkIGINVidiaDataSourceGui.h"

class QmitkIGINVidiaDataSource;
class QLabel;
class QGridLayout;
class QmitkVideoPreviewWidget;

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

  void StopPreviewWidget();

protected slots:

  /**
   * Directly called by data-source-gui-manager (I think...) whenever this datasource GUI should refresh.
   */
  virtual void Update();

  void OnFieldModeChange(int index);

private:

  QmitkIGINVidiaDataSource* GetQmitkIGINVidiaDataSource() const;

  // init'd by Initialize()
  QmitkVideoPreviewWidget* m_OglWin;

  // used to check whether stream format has changed in between calls to OnUpdateDisplay()
  int     m_PreviousBaseResolution;
}; // end class

#endif

