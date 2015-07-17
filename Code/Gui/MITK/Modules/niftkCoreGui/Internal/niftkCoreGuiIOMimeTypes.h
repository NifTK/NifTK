/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __niftkCoreGuiIOMimeTypes_h
#define __niftkCoreGuiIOMimeTypes_h

#include <mitkCustomMimeType.h>
#include <string>

namespace niftk {

class CoreGuiIOMimeTypes
{
public:

  static std::vector<mitk::CustomMimeType*> Get();

  static mitk::CustomMimeType LABELMAP_MIMETYPE();
  static std::string LABELMAP_MIMETYPE_NAME();
  static std::string LABELMAP_MIMETYPE_DESCRIPTION();
  
private:

  CoreGuiIOMimeTypes(); // purposely not implemented
  CoreGuiIOMimeTypes(const CoreGuiIOMimeTypes&); // purposely not implemented
};

} // end namespace

#endif
