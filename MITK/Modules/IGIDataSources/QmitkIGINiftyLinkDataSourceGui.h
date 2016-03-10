/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkIGINiftyLinkDataSourceGui_h
#define QmitkIGINiftyLinkDataSourceGui_h

#include "niftkIGIDataSourcesExports.h"
#include "QmitkIGIDataSourceGui.h"
#include "QmitkIGINiftyLinkDataSource.h"

class ClientDescriptorXMLBuilder;

/**
 * \class QmitkIGINiftyLinkDataSourceGui
 * \brief Base class for NiftyLink Data Source Guis.
 */
class NIFTKIGIDATASOURCES_EXPORT QmitkIGINiftyLinkDataSourceGui : public QmitkIGIDataSourceGui
{
  Q_OBJECT

public:

  mitkClassMacro(QmitkIGINiftyLinkDataSourceGui, QmitkIGIDataSourceGui);

  /**
   * \brief Initializes this widget.
   */
  virtual void Initialize(QWidget *parent);

protected:

  QmitkIGINiftyLinkDataSourceGui(); // Purposefully hidden.
  virtual ~QmitkIGINiftyLinkDataSourceGui(); // Purposefully hidden.

  QmitkIGINiftyLinkDataSourceGui(const QmitkIGINiftyLinkDataSourceGui&); // Purposefully not implemented.
  QmitkIGINiftyLinkDataSourceGui& operator=(const QmitkIGINiftyLinkDataSourceGui&); // Purposefully not implemented.

  /**
   * \brief Gets the source attached to this object.
   */
  virtual QmitkIGINiftyLinkDataSource* GetQmitkIGINiftyLinkDataSource();

  /**
   * \brief Called by Initialize(QWidget *parent), where either parent or config could be null.
   */
  virtual void Initialize(QWidget *parent, niftk::NiftyLinkClientDescriptor* config) = 0;

}; // end class

#endif
