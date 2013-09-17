/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkIGIUltrasonixToolGui_h
#define QmitkIGIUltrasonixToolGui_h

#include "niftkIGIGuiExports.h"
#include "QmitkIGINiftyLinkDataSourceGui.h"
#include "ui_QmitkIGIUltrasonixToolGui.h"
#include <NiftyLinkMessage.h>
#include <mitkImage.h>

class QmitkIGIUltrasonixTool;
class ClientDescriptorXMLBuilder;
class QImage;
class QLabel;

/**
 * \class QmitkIGIUltrasonixToolGui
 * \brief Implements a tool GUI interface to receive and process messages from the Ultrasonix scanner.
 */
class NIFTKIGIGUI_EXPORT QmitkIGIUltrasonixToolGui : public QmitkIGINiftyLinkDataSourceGui, public Ui_QmitkIGIUltrasonixToolGui
{
  Q_OBJECT

public:

  mitkClassMacro(QmitkIGIUltrasonixToolGui, QmitkIGINiftyLinkDataSourceGui);
  itkNewMacro(QmitkIGIUltrasonixToolGui);

  /**
   * \brief Retrieves data from the source, to update the GUI display.
   */
  virtual void Update();

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

  /**
   * \brief Sets up which image to follow.
   */
  void InitializeImage();

private:

  QmitkIGIUltrasonixTool *m_UltrasonixTool;
  mitk::Image            *m_Image;
}; // end class

#endif

