/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkGeneralSegmentorPipelineCache.h"

namespace niftk
{

//-----------------------------------------------------------------------------
GeneralSegmentorPipelineCache::GeneralSegmentorPipelineCache()
{
}


//-----------------------------------------------------------------------------
GeneralSegmentorPipelineCache::~GeneralSegmentorPipelineCache()
{
  std::map<std::string, GeneralSegmentorPipelineInterface*>::iterator it = m_TypeToPipelineMap.begin();
  std::map<std::string, GeneralSegmentorPipelineInterface*>::iterator itEnd = m_TypeToPipelineMap.end();

  while (it != itEnd)
  {
    delete it->second;
  }
}


//-----------------------------------------------------------------------------
GeneralSegmentorPipelineCache* GeneralSegmentorPipelineCache::Instance()
{
  static GeneralSegmentorPipelineCache* s_instance = nullptr;
  if (!s_instance)
  {
    s_instance = new GeneralSegmentorPipelineCache();
  }
  return s_instance;
}

}
