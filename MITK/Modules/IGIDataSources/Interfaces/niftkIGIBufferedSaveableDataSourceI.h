/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
#ifndef niftkIGIBufferedSaveableDataSourceI_h
#define niftkIGIBufferedSaveableDataSourceI_h

#include "niftkIGIDataSourcesExports.h"
#include <niftkIGIDataType.h>

namespace niftk
{

/**
* \brief Abstract base class for data sources that can clean their own buffer.
*
* Note: All errors should thrown as mitk::Exception or sub-classes thereof.
*/
class NIFTKIGIDATASOURCES_EXPORT IGIBufferedSaveableDataSourceI
{
public:

  virtual void SaveItem(niftk::IGIDataType& item) = 0;

protected:

  IGIBufferedSaveableDataSourceI() {} // Purposefully hidden.
  virtual ~IGIBufferedSaveableDataSourceI() {} // Purposefully hidden.

  IGIBufferedSaveableDataSourceI(const IGIBufferedSaveableDataSourceI&); // Purposefully not implemented.
  IGIBufferedSaveableDataSourceI& operator=(const IGIBufferedSaveableDataSourceI&); // Purposefully not implemented.

};

} // end namespace

#endif
