/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

//-----------------------------------------------------------------------------
template<class TInputImage, class TOutputImage, class TPointSet>
MIDASThresholdingRegionGrowingImageFilter<TInputImage, TOutputImage, TPointSet>::MIDASThresholdingRegionGrowingImageFilter()
  : m_LowerThreshold(0),
    m_UpperThreshold(0)
{
}

//-----------------------------------------------------------------------------
template<class TInputImage, class TOutputImage, class TPointSet>
void MIDASThresholdingRegionGrowingImageFilter<TInputImage, TOutputImage, TPointSet>::ConditionalAddPixel(
    std::stack<typename OutputImageType::IndexType>& r_stack,
    const typename OutputImageType::IndexType& currentImgIdx,
    const typename OutputImageType::IndexType& nextImgIdx,
    bool isFullyConnected
    )
{
  /// I.e. out of thresholds.
  InputPixelType inputImageNextPixel = this->GetInput()->GetPixel(nextImgIdx);
  if (inputImageNextPixel < m_LowerThreshold || inputImageNextPixel > m_UpperThreshold)
  {
    return;
  }

  Superclass::ConditionalAddPixel(r_stack, currentImgIdx, nextImgIdx, isFullyConnected);
}
