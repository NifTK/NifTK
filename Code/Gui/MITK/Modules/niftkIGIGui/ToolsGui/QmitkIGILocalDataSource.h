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

public slots:

signals:

protected:

  QmitkIGILocalDataSource(); // Purposefully hidden.
  virtual ~QmitkIGILocalDataSource(); // Purposefully hidden.

  QmitkIGILocalDataSource(const QmitkIGILocalDataSource&); // Purposefully not implemented.
  QmitkIGILocalDataSource& operator=(const QmitkIGILocalDataSource&); // Purposefully not implemented.

private:

}; // end class

#endif

