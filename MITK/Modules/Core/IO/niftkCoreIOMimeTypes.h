/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __niftkCoreIOMimeTypes_h
#define __niftkCoreIOMimeTypes_h

#include <mitkCustomMimeType.h>
#include <string>

namespace niftk 
{

class CoreIOMimeTypes
{

public:

  static std::vector<mitk::CustomMimeType*> Get();
  
  static mitk::CustomMimeType LABELMAP_MIMETYPE();
  static std::string LABELMAP_MIMETYPE_NAME();
  static std::string LABELMAP_MIMETYPE_DESCRIPTION();
  
private:

  CoreIOMimeTypes(); // purposely not implemented
  CoreIOMimeTypes(const CoreIOMimeTypes&); // purposely not implemented
};

}

#endif
