/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkCoreIOMimeTypes.h"
#include <mitkIOMimeTypes.h>

namespace niftk
{

//-----------------------------------------------------------------------------
std::vector<mitk::CustomMimeType*> CoreIOMimeTypes::Get()
{
  std::vector<mitk::CustomMimeType*> mimeTypes;

  // order matters here (descending rank for mime types)
  mimeTypes.push_back(TRANSFORM4X4_MIMETYPE().Clone());

  return mimeTypes;
}


//-----------------------------------------------------------------------------
mitk::CustomMimeType CoreIOMimeTypes::TRANSFORM4X4_MIMETYPE()
{
  std::string category = "4x4 Transform File";

  mitk::CustomMimeType mimeType(TRANSFORM4X4_MIMETYPE_NAME());
  mimeType.SetCategory(category);
  mimeType.SetComment("4x4 Transform");
  mimeType.AddExtension("4x4");
  return mimeType;
}


//-----------------------------------------------------------------------------
std::string CoreIOMimeTypes::TRANSFORM4X4_MIMETYPE_NAME()
{
  return mitk::IOMimeTypes::DEFAULT_BASE_NAME() + ".4x4";
}


//-----------------------------------------------------------------------------
std::string CoreIOMimeTypes::TRANSFORM4X4_MIMETYPE_DESCRIPTION()
{
  return "4x4 Transforms";
}

} // end namespace
