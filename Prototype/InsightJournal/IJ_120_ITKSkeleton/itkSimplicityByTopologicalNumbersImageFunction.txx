#ifndef itkSimplicityByTopologicalNumbersImageFunction_txx
#define itkSimplicityByTopologicalNumbersImageFunction_txx

#include "itkSimplicityByTopologicalNumbersImageFunction.h"

namespace itk
{

template<typename TImage, typename TForegroundConnectivity, 
         typename TBackgroundConnectivity >
SimplicityByTopologicalNumbersImageFunction<TImage, TForegroundConnectivity, 
                                            TBackgroundConnectivity>
::SimplicityByTopologicalNumbersImageFunction()
  {
  m_TnCounter = TopologicalNumberImageFunction<TImage, 
                  TForegroundConnectivity>::New();
  }

template<typename TImage, typename TForegroundConnectivity, 
        typename TBackgroundConnectivity >
bool
SimplicityByTopologicalNumbersImageFunction<TImage, TForegroundConnectivity, 
                                            TBackgroundConnectivity>
::Evaluate(PointType const & point) const
  {
  typename TImage::IndexType index;
  ConvertPointToNearestIndex(point, index);
  return EvaluateAtIndex(index);
  }


template<typename TImage, typename TForegroundConnectivity, 
         typename TBackgroundConnectivity >
bool
SimplicityByTopologicalNumbersImageFunction<TImage, TForegroundConnectivity, 
                                            TBackgroundConnectivity>
::EvaluateAtIndex(IndexType const & index) const
  {
  std::pair<unsigned char, unsigned char> const result = 
    m_TnCounter->EvaluateAtIndex(index);
  return (result.first==1 && result.second==1);
  }


template<typename TImage, typename TForegroundConnectivity, 
         typename TBackgroundConnectivity >
bool
SimplicityByTopologicalNumbersImageFunction<TImage, TForegroundConnectivity, 
                                            TBackgroundConnectivity>
::EvaluateAtContinuousIndex(ContinuousIndexType const & contIndex) const
  {
  typename TImage::IndexType index;
  ConvertContinuousIndexToNearestIndex(contIndex, index);
  return EvaluateAtIndex(index);
  }

}

#endif // itkSimplicityByTopologicalNumbersImageFunction_txx
