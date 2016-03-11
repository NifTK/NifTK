/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __niftkGeneralSegmentorPipelineCache_h
#define __niftkGeneralSegmentorPipelineCache_h

#include "niftkMIDASExports.h"

#include "niftkGeneralSegmentorPipeline.h"

namespace niftk
{

class GeneralSegmentorPipelineInterface;

/// \class GeneralSegmentorPipelineCache
class NIFTKMIDAS_EXPORT GeneralSegmentorPipelineCache
{
public:

  static GeneralSegmentorPipelineCache* Instance();

  template<typename TPixel, unsigned int VImageDimension>
  GeneralSegmentorPipeline<TPixel, VImageDimension>* GetPipeline();

  template<typename TPixel, unsigned int VImageDimension>
  void DestroyPipeline();

protected:
  GeneralSegmentorPipelineCache();
  virtual ~GeneralSegmentorPipelineCache();

private:

  /// \brief We hold a Map, containing a key comprised of the "typename TPixel, unsigned int VImageDimension"
  /// as a key, and the object containing the whole pipeline for single slice 2D region growing.
  typedef std::pair<std::string, GeneralSegmentorPipelineInterface*> StringAndPipelineInterfacePair;
  std::map<std::string, GeneralSegmentorPipelineInterface*> m_TypeToPipelineMap;

};

}

#include "niftkGeneralSegmentorPipelineCache.txx"

#endif
