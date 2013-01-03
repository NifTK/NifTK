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
 #include "QmitkIGINiftyLinkDataSourceGui.h"
#include "ui_QmitkIGIUltrasonixToolGui.h"
#include <OIGTLMessage.h>

class QmitkIGIUltrasonixTool;
class ClientDescriptorXMLBuilder;
class QImage;
class QLabel;

/**
 * \class QmitkIGITrackerToolGui
 * \brief Implements a tool GUI interface to receive and process messages from the Ultrasonix scanner.
 */
class NIFTKIGIGUI_EXPORT QmitkIGIUltrasonixToolGui : public QmitkIGINiftyLinkDataSourceGui, public Ui_QmitkIGIUltrasonixToolGui
{
  Q_OBJECT

public:

  mitkClassMacro(QmitkIGIUltrasonixToolGui, QmitkIGINiftyLinkDataSourceGui);
  itkNewMacro(QmitkIGIUltrasonixToolGui);

protected:

  QmitkIGIUltrasonixToolGui(); // Purposefully hidden.
  virtual ~QmitkIGIUltrasonixToolGui(); // Purposefully hidden.

  QmitkIGIUltrasonixToolGui(const QmitkIGIUltrasonixToolGui&); // Purposefully not implemented.
  QmitkIGIUltrasonixToolGui& operator=(const QmitkIGIUltrasonixToolGui&); // Purposefully not implemented.

  /**
   * \brief Initializes this widget, calling Ui_QmitkIGIUltrasonixToolGui::setupUi(parent),
   * and any other stuff as necessary.
   */
  virtual void Initialize(QWidget *parent, ClientDescriptorXMLBuilder *config);

private slots:

  void OnStatusUpdate(QString message);
  void OnUpdatePreviewDisplay(QImage* image, float motorPosition);

private:

  QmitkIGIUltrasonixTool* GetQmitkIGIUltrasonixTool() const;
  QLabel *m_PixmapLabel;

}; // end class

#endif

