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
#include "QmitkIGINiftyLinkDataSourceGui.h"
#include "ui_QmitkIGIUltrasonixToolGui.h"
#include <NiftyLinkMessage.h>

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

