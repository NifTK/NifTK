/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
niftk::GeneralSegmentorPipeline<TPixel, VImageDimension>* niftk::GeneralSegmentorPipelineCache::GetPipeline()
{
  std::stringstream key;
  key << typeid(TPixel).name() << VImageDimension;

  GeneralSegmentorPipeline<TPixel, VImageDimension>* pipeline = NULL;
  GeneralSegmentorPipelineInterface* myPipeline = NULL;

  std::map<std::string, GeneralSegmentorPipelineInterface*>::iterator it;
  it = m_TypeToPipelineMap.find(key.str());

  if (it == m_TypeToPipelineMap.end())
  {
    pipeline = new niftk::GeneralSegmentorPipeline<TPixel, VImageDimension>();
    myPipeline = pipeline;
    m_TypeToPipelineMap.insert(StringAndPipelineInterfacePair(key.str(), pipeline));
  }
  else
  {
    pipeline = static_cast<niftk::GeneralSegmentorPipeline<TPixel, VImageDimension>*>(it->second);
  }

  return pipeline;
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void niftk::GeneralSegmentorPipelineCache::DestroyPipeline()
{
  std::stringstream key;
  key << typeid(TPixel).name() << VImageDimension;

  std::map<std::string, GeneralSegmentorPipelineInterface*>::iterator it;
  it = m_TypeToPipelineMap.find(key.str());

  delete it->second;

  m_TypeToPipelineMap.erase(it);
}
