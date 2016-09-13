/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkIOUtil_h
#define niftkIOUtil_h

#include "niftkCoreExports.h"

#include <niftkLookupTableProviderService.h>

namespace niftk
{

class NIFTKCORE_EXPORT IOUtil
{
  typedef IOUtil Self;

  IOUtil();

  ~IOUtil();

  static LookupTableProviderService* GetLookupTableProviderService();

public:

  /// \brief Attempts to load LookupTable from given file, returning display name of LookupTable if successful.
  static QString LoadLookupTable(QString& fileName);

};

}

#endif
