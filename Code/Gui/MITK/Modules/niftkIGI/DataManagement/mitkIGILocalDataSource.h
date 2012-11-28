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

#ifndef MITKIGILOCALDATASOURCE_H
#define MITKIGILOCALDATASOURCE_H

#include "niftkIGIExports.h"
#include "mitkIGIDataSource.h"

namespace mitk
{

/**
 * \class QmitkIGILocalDataSource
 * \brief Base class for IGI Data Sources that are not receiving networked input,
 * and hence are grabbing data from the local machine - eg. Video grabber.
 */
class NIFTKIGI_EXPORT IGILocalDataSource : public IGIDataSource
{

public:

  mitkClassMacro(IGILocalDataSource, IGIDataSource);

protected:

  IGILocalDataSource(); // Purposefully hidden.
  virtual ~IGILocalDataSource(); // Purposefully hidden.

  IGILocalDataSource(const IGILocalDataSource&); // Purposefully not implemented.
  IGILocalDataSource& operator=(const IGILocalDataSource&); // Purposefully not implemented.

private:

}; // end class

} // end namespace

#endif

