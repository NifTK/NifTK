/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkCoreGuiIOMimeTypes.h"
#include <mitkIOMimeTypes.h>

namespace niftk
{

//-----------------------------------------------------------------------------
std::vector<mitk::CustomMimeType*> CoreGuiIOMimeTypes::Get()
{
  std::vector<mitk::CustomMimeType*> mimeTypes;

  // order matters here (descending rank for mime types)
  mimeTypes.push_back(LABELMAP_MIMETYPE().Clone());

  return mimeTypes;
}


// ----------------------------------------------------------------
// LabelMap Mime type
mitk::CustomMimeType CoreGuiIOMimeTypes::LABELMAP_MIMETYPE()
{
  mitk::CustomMimeType mimeType(LABELMAP_MIMETYPE_NAME());
  std::string category = "LabelMap File";
  mimeType.SetComment("EpiNav Label Map format");
  mimeType.SetCategory(category);
  mimeType.AddExtension("lmap");
    
  return mimeType;
}

std::string CoreGuiIOMimeTypes::LABELMAP_MIMETYPE_NAME()
{
  // Have to pick one that becomes the default. xmlE is the least restrictive.
  static std::string name = mitk::IOMimeTypes::DEFAULT_BASE_NAME() + ".lmap";
  return name;
}

std::string CoreGuiIOMimeTypes::LABELMAP_MIMETYPE_DESCRIPTION()
{
  static std::string description = "EpiNav label map that defines the mapping between color and pixel value.";
  return description;
}

} // end namespace
