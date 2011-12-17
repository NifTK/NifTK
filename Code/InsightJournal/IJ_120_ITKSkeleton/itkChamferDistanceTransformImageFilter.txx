#ifndef itkChamferDistanceTransformImageFilter_txx
#define itkChamferDistanceTransformImageFilter_txx

#include <algorithm>
#include <vector>

#include <itkConstantBoundaryCondition.h>
#include <itkImageRegionIterator.h>
#include <itkImageRegionConstIterator.h>
#include <itkNeighborhood.h>
#include <itkNeighborhoodIterator.h>
#include <itkNumericTraits.h>

#include "itkChamferDistanceTransformImageFilter.h"

namespace itk
{

template<typename InputImage, typename OutputImage>
ChamferDistanceTransformImageFilter<InputImage, OutputImage>
::ChamferDistanceTransformImageFilter()
: m_DistanceFromObject(true)
  {
  std::fill(m_Weights, m_Weights+OutputImage::ImageDimension, 1);
  }


template<typename InputImage, typename OutputImage>
template<typename Iterator>
void 
ChamferDistanceTransformImageFilter<InputImage, OutputImage>
::SetWeights(Iterator begin, Iterator end)
  {
  std::copy(begin, end, m_Weights);
  }


template<typename InputImage, typename OutputImage>
std::vector<typename OutputImage::PixelType>
ChamferDistanceTransformImageFilter<InputImage, OutputImage>
::GetWeights() const
  {
  std::vector<typename OutputImage::PixelType> 
    result(OutputImage::ImageDimension);
  std::copy(m_Weights, m_Weights+OutputImage::ImageDimension, result.begin());
  
  return result;
  }

template<typename InputImage, typename OutputImage>
void 
ChamferDistanceTransformImageFilter<InputImage, OutputImage>
::PrintSelf(std::ostream& os, Indent indent) const
  {
  Superclass::PrintSelf(os, indent);
  os << indent << "Weights : [ ";
  for(unsigned int i=0; i< OutputImage::ImageDimension; ++i)
    {
    os << m_Weights[i] << " ";
    }
  os << "]" << "\n";
  os << indent << "Distance from object : " << m_DistanceFromObject << "\n";
  }


template<typename InputImage, typename OutputImage>
void
ChamferDistanceTransformImageFilter<InputImage, OutputImage>
::GenerateData()
  {
  // Allocate output image
  typename OutputImageType::RegionType region;
  typename OutputImageType::RegionType::IndexType start;
  start.Fill(0);
  region.SetIndex(start);
  typename OutputImageType::RegionType::SizeType 
    size( this->GetInput()->GetRequestedRegion().GetSize() );
  region.SetSize(size);
  
  typename OutputImageType::Pointer outputImage = this->GetOutput();
  outputImage->SetRegions(region);
  outputImage->Allocate();

  // Initialize output image : \infty where input is not 0, 0 where input is 0 
  // if computing the distance from the object, the opposite otherwise.
  typename OutputImageType::PixelType const fgValue = 
    m_DistanceFromObject?
      0:NumericTraits<typename OutputImageType::PixelType>::max();
  typename OutputImageType::PixelType const bgValue = 
    NumericTraits<typename OutputImageType::PixelType>::max() - fgValue;
  
  itk::ImageRegionConstIterator<InputImageType> 
    inputImageIt(this->GetInput(0), this->GetInput()->GetRequestedRegion());
  itk::ImageRegionIterator<OutputImageType> 
    outputImageIt(this->GetOutput(), this->GetOutput()->GetRequestedRegion());
  while(!outputImageIt.IsAtEnd())
    {
    typename OutputImageType::PixelType const value = 
      (inputImageIt.Get() != NumericTraits<typename InputImageType::PixelType>::Zero) ? 
        fgValue : bgValue;
    outputImageIt.Set(value);
    
    ++inputImageIt;
    ++outputImageIt;
    }
    
  // Create the mask of the weights
  Neighborhood<typename OutputImageType::PixelType, 
               OutputImageType::ImageDimension> mask;
  mask.SetRadius(1);
  for(unsigned int i=0; i<mask.Size(); ++i)
    {
    // Skip center
    if(i == mask.Size()/2)
      {
      mask[mask.Size()/2] = 
        NumericTraits<typename OutputImageType::PixelType>::max();
      continue;
      }
    
    Offset<OutputImageType::ImageDimension> const offset = mask.GetOffset(i);
    int type=-1;
    for(unsigned int j=0; j<OutputImageType::ImageDimension; ++j)
      {
      if(offset[j]!=0)
        {
        ++type;
        }
      }
    mask[i] = m_Weights[type];
  }
    
    // Prepare the neighborhood iterator
    typename NeighborhoodIterator<InputImageType>::RadiusType r;
    r.Fill(1);
    NeighborhoodIterator<OutputImageType> 
      it(r, this->GetOutput(), this->GetOutput()->GetRequestedRegion());
    ConstantBoundaryCondition<OutputImageType> bc;
    bc.SetConstant(bgValue);
    it.OverrideBoundaryCondition(&bc);
    unsigned int const maskCenter = it.Size()/2;
    

    // First pass : forward scan, use backward mask
    it.GoToBegin();
    
    while(!it.IsAtEnd())
      {
      typename OutputImageType::PixelType minimum = 
        NumericTraits<typename OutputImageType::PixelType>::max();
      for(unsigned int i=0; i<maskCenter; ++i)
        {
        if(it.GetPixel(i) < 
           NumericTraits<typename OutputImageType::PixelType>::max() - mask[i])
          {
          minimum = std::min(minimum, it.GetPixel(i) + mask[i]);
          }
        }
      if(minimum < it.GetCenterPixel())
        {
        it.SetCenterPixel(minimum);
        }
      
      
      ++it;
      }
    
    // Second pass : bacward scan, use forward mask
    it.GoToEnd();
    --it;
    while(!it.IsAtBegin())
      {
      typename OutputImageType::PixelType minimum = 
        NumericTraits<typename OutputImageType::PixelType>::max();
      for(unsigned int i=maskCenter+1; i<it.Size(); ++i)
        {
        if(it.GetPixel(i) < 
           NumericTraits<typename OutputImageType::PixelType>::max() - mask[i])
          {
          minimum = std::min(minimum, it.GetPixel(i) + mask[i]);
          }
        }
      if(minimum < it.GetCenterPixel())
        {
        it.SetCenterPixel(minimum);
        }
      --it;
      }
    // Last pixel in the scan.
    typename OutputImageType::PixelType minimum = 
      NumericTraits<typename OutputImageType::PixelType>::max();
    for(unsigned int i=maskCenter+1; i<it.Size(); ++i)
      {
      if(it.GetPixel(i) < NumericTraits<typename OutputImageType::PixelType>::max() - mask[i])
        {
        minimum = std::min(minimum, it.GetPixel(i) + mask[i]);
        }
      }
    if(minimum < it.GetCenterPixel())
      {
      it.SetCenterPixel(minimum);
      }
    
  }

}

#endif // itkChamferDistanceTransformImageFilter_txx
