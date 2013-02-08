/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QMITKIGILOCALDATASOURCE_H
#define QMITKIGILOCALDATASOURCE_H

#include "niftkIGIGuiExports.h"
#include "QmitkIGIDataSource.h"

/**
 * \class QmitkIGILocalDataSource
 * \brief Base class for IGI Data Sources that are not receiving networked input,
 * and hence are grabbing data from the local machine - eg. Video grabber.
 */
class NIFTKIGIGUI_EXPORT QmitkIGILocalDataSource : public QmitkIGIDataSource
{

  Q_OBJECT

public:

  mitkClassMacro(QmitkIGILocalDataSource, QmitkIGIDataSource);

protected:

  QmitkIGILocalDataSource(); // Purposefully hidden.
  virtual ~QmitkIGILocalDataSource(); // Purposefully hidden.

  QmitkIGILocalDataSource(const QmitkIGILocalDataSource&); // Purposefully not implemented.
  QmitkIGILocalDataSource& operator=(const QmitkIGILocalDataSource&); // Purposefully not implemented.

private:

}; // end class

#endif

