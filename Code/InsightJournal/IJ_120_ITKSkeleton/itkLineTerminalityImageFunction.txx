#ifndef itkLineTerminalityImageFunction_txx
#define itkLineTerminalityImageFunction_txx

#include "itkLineTerminalityImageFunction.h"

namespace itk
{

template<typename TImage, typename TForegroundConnectivity, 
         typename TBackgroundConnectivity >
LineTerminalityImageFunction<TImage, TForegroundConnectivity, 
                             TBackgroundConnectivity>
::LineTerminalityImageFunction()
  {
  }

template<typename TImage, typename TForegroundConnectivity, 
         typename TBackgroundConnectivity >
bool
LineTerminalityImageFunction<TImage, TForegroundConnectivity, 
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
LineTerminalityImageFunction<TImage, TForegroundConnectivity, 
                             TBackgroundConnectivity>
::EvaluateAtIndex(IndexType const & index) const
  {
  TForegroundConnectivity const & fgc = TForegroundConnectivity::GetInstance();
  int nbNeighbors = 0;
  for(int i=0; i<fgc.GetNumberOfNeighbors() && nbNeighbors<=1; ++i)
    {
    itk::Offset<TForegroundConnectivity::Dimension> offset;
    for(unsigned int j=0; j<TForegroundConnectivity::Dimension; ++j)
      {
      offset[j] = fgc.GetNeighborsPoints()[i][j];
      }
    if(this->GetInputImage()->GetPixel(index+offset)!=
       NumericTraits<typename TImage::PixelType>::Zero)
      {
      ++nbNeighbors;
      }
  }
  return (nbNeighbors==1);
  }


template<typename TImage, typename TForegroundConnectivity, 
         typename TBackgroundConnectivity >
bool
LineTerminalityImageFunction<TImage, TForegroundConnectivity, 
                             TBackgroundConnectivity>
::EvaluateAtContinuousIndex(ContinuousIndexType const & contIndex) const
{
  typename TImage::IndexType index;
  ConvertContinuousIndexToNearestIndex(contIndex, index);
  return EvaluateAtIndex(index);
}

}

#endif // itkLineTerminalityImageFunction_txx
