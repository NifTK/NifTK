/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkIGIMatrixPerFileBackend_h
#define niftkIGIMatrixPerFileBackend_h

#include <niftkIGITrackersExports.h>
#include "niftkIGITrackerBackend.h"

namespace niftk
{
class NIFTKIGITRACKERS_EXPORT IGIMatrixPerFileBackend : public niftk::IGITrackerBackend
{
public:

  mitkClassMacroItkParent(IGIMatrixPerFileBackend, niftk::IGITrackerBackend)

protected:

  IGIMatrixPerFileBackend(mitk::DataStorage::Pointer dataStorage); // Purposefully hidden.
  virtual ~IGIMatrixPerFileBackend(); // Purposefully hidden.

  IGIMatrixPerFileBackend(const IGIMatrixPerFileBackend&); // Purposefully not implemented.
  IGIMatrixPerFileBackend& operator=(const IGIMatrixPerFileBackend&); // Purposefully not implemented.

};

} // end namespace

#endif
