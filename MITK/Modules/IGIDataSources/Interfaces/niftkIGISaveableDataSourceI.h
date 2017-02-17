/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
#ifndef niftkIGISaveableDataSourceI_h
#define niftkIGISaveableDataSourceI_h

#include "niftkIGIDataSourcesExports.h"

namespace niftk
{

/**
* \brief Abstract base class for data sources that can save their own buffer.
*
* Note: All errors should thrown as mitk::Exception or sub-classes thereof.
*/
class NIFTKIGIDATASOURCES_EXPORT IGISaveableDataSourceI
{
public:

  virtual void SaveBuffer() = 0;

protected:

  IGISaveableDataSourceI() {} // Purposefully hidden.
  virtual ~IGISaveableDataSourceI() {} // Purposefully hidden.

  IGISaveableDataSourceI(const IGISaveableDataSourceI&); // Purposefully not implemented.
  IGISaveableDataSourceI& operator=(const IGISaveableDataSourceI&); // Purposefully not implemented.

};

} // end namespace

#endif
