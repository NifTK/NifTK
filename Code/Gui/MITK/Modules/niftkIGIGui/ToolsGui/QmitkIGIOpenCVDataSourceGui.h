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

#ifndef QMITKIGIULTRASONIXTOOLGUI_H
#define QMITKIGIULTRASONIXTOOLGUI_H

#include "niftkIGIGuiExports.h"
#include <QWidget>
#include <mitkVideoSource.h>
#include <mitkRenderingManager.h>
#include "QmitkIGIDataSourceGui.h"

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

  QmitkRenderWindow               *m_RenderWindow;
  mitk::RenderingManager::Pointer  m_RenderingManager;
  QGridLayout                     *m_Layout;

}; // end class

#endif

