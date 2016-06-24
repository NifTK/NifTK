/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
#ifndef niftkIGICleanableDataSourceI_h
#define niftkIGICleanableDataSourceI_h

#include "niftkIGIDataSourcesExports.h"

namespace niftk
{

/**
* \brief Abstract base class for data sources that can clean their own buffer.
*
* Note: All errors should thrown as mitk::Exception or sub-classes thereof.
*/
class NIFTKIGIDATASOURCES_EXPORT IGICleanableDataSourceI
{
public:

  virtual void CleanBuffer() = 0;

protected:

  IGICleanableDataSourceI(); // Purposefully hidden.
  virtual ~IGICleanableDataSourceI(); // Purposefully hidden.

  IGICleanableDataSourceI(const IGICleanableDataSourceI&); // Purposefully not implemented.
  IGICleanableDataSourceI& operator=(const IGICleanableDataSourceI&); // Purposefully not implemented.

};

} // end namespace

#endif
