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

#ifndef QMITKIGINIFTYLINKDATASOURCEGUI_H
#define QMITKIGINIFTYLINKDATASOURCEGUI_H

#include "niftkIGIGuiExports.h"
#include "QmitkIGIDataSourceGui.h"
#include "QmitkIGITrackerTool.h"
#include "ui_QmitkIGITrackerToolGui.h"

class ClientDescriptorXMLBuilder;

/**
 * \class QmitkIGINiftyLinkDataSourceGui
 * \brief Base class for NiftyLink Data Source Guis.
 */
class NIFTKIGIGUI_EXPORT QmitkIGINiftyLinkDataSourceGui : public QmitkIGIDataSourceGui
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
  virtual void Initialize(QWidget *parent, ClientDescriptorXMLBuilder* config) = 0;

}; // end class

#endif

