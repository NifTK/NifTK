/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkIGITrackerSourceGui_h
#define QmitkIGITrackerSourceGui_h

#include "niftkIGIGuiExports.h"
#include "QmitkIGINiftyLinkDataSourceGui.h"
#include "ui_QmitkIGITrackerSourceGui.h"

class ClientDescriptorXMLBuilder;
class QmitkIGITrackerSource;

/**
 * \class QmitkIGITrackerSourceGui
 * \brief Base class for IGI Tracker Source GUIs.
 */
class NIFTKIGIGUI_EXPORT QmitkIGITrackerSourceGui : public QmitkIGINiftyLinkDataSourceGui, public Ui_QmitkIGITrackerSourceGui
{
  Q_OBJECT

public:

  mitkClassMacro(QmitkIGITrackerSourceGui, QmitkIGINiftyLinkDataSourceGui);
  itkNewMacro(QmitkIGITrackerSourceGui);

protected:

  QmitkIGITrackerSourceGui(); // Purposefully hidden.
  virtual ~QmitkIGITrackerSourceGui(); // Purposefully hidden.

  QmitkIGITrackerSourceGui(const QmitkIGITrackerSourceGui&); // Purposefully not implemented.
  QmitkIGITrackerSourceGui& operator=(const QmitkIGITrackerSourceGui&); // Purposefully not implemented.

  /**
   * \brief Initializes this widget, calling Ui_QmitkIGITrackerSourceGui::setupUi(parent),
   * and any other stuff as necessary.
   */
  virtual void Initialize(QWidget *parent, ClientDescriptorXMLBuilder* config);

private slots:

  /**
   * \brief Used to update the GUI with a NiftyLink tracking message.
   */
  void OnStatusUpdate(QString message);

private:

  QmitkIGITrackerSource* GetQmitkIGITrackerSource() const;
}; // end class

#endif

