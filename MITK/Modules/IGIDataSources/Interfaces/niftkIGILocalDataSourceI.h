/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
#ifndef niftkIGILocalDataSourceI_h
#define niftkIGILocalDataSourceI_h

#include "niftkIGIDataSourcesExports.h"

namespace niftk
{

/**
* \brief Abstract base class for local data sources.
*
* (Ones that grab data themselves, as opposed to, for example, receive via OpenIGTLink)
*
* Note: All errors should thrown as mitk::Exception or sub-classes thereof.
*/
class NIFTKIGIDATASOURCES_EXPORT IGILocalDataSourceI
{
public:

  virtual void GrabData() = 0;

protected:

  IGILocalDataSourceI() {} // Purposefully hidden.
  virtual ~IGILocalDataSourceI() {} // Purposefully hidden.

  IGILocalDataSourceI(const IGILocalDataSourceI&); // Purposefully not implemented.
  IGILocalDataSourceI& operator=(const IGILocalDataSourceI&); // Purposefully not implemented.

};

} // end namespace

#endif
