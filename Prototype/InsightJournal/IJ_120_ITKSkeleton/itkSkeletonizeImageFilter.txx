#ifndef itkSkeletonizationImageFilter_txx
#define itkSkeletonizationImageFilter_txx

#include <functional>

#include <itkImageRegionConstIteratorWithIndex.h>
#include <itkNumericTraits.h>
#include <itkProgressReporter.h>

#include "itkHierarchicalQueue.h"

#include "itkLineTerminalityImageFunction.h"
#include "itkSimplicityByTopologicalNumbersImageFunction.h"

#include "itkSkeletonizeImageFilter.h"

namespace itk
{

template<typename TImage, typename TForegroundConnectivity>
void
SkeletonizeImageFilter<TImage, TForegroundConnectivity>
::SetOrderingImage(OrderingImageType *input)
  {
  // Process object is not const-correct so the const casting is required.
  this->SetNthInput( 1, const_cast<OrderingImageType *>(input) );
  }

template<typename TImage, typename TForegroundConnectivity>
typename 
  SkeletonizeImageFilter<TImage, TForegroundConnectivity>::OrderingImageType * 
SkeletonizeImageFilter<TImage, TForegroundConnectivity>
::GetOrderingImage()
  {
  return static_cast<OrderingImageType*>(
    const_cast<DataObject *>(this->ProcessObject::GetInput(1))
  );
  }

template<typename TImage, typename TForegroundConnectivity>
SkeletonizeImageFilter<TImage, TForegroundConnectivity>
::SkeletonizeImageFilter()
: m_SimplicityCriterion(0),
  m_TerminalityCriterion(0)
  {
  this->SetNumberOfRequiredInputs(2);
  }


template<typename TImage, typename TForegroundConnectivity>
void 
SkeletonizeImageFilter<TImage, TForegroundConnectivity>
::PrintSelf(std::ostream& os, Indent indent) const
  {
  Superclass::PrintSelf(os,indent);
  os << indent 
     << "Cell dimension used for foreground connectivity: " 
     <<  ForegroundConnectivity::CellDimension << std::endl;
  }


template<typename TImage, typename TForegroundConnectivity>
void 
SkeletonizeImageFilter<TImage, TForegroundConnectivity>
::GenerateInputRequestedRegion()
  {
  // call the superclass' implementation of this method
  Superclass::GenerateInputRequestedRegion();

  // get pointers to the inputs
  typename OrderingImageType::Pointer orderingPtr = this->GetOrderingImage();

  typename TImage::Pointer inputPtr = const_cast<TImage*>(this->GetInput());

  if ( !orderingPtr || !inputPtr )
    { 
    return; 
    }

  // We need to
  // configure the inputs such that all the data is available.
  //
  orderingPtr->SetRequestedRegion(orderingPtr->GetLargestPossibleRegion());
  inputPtr->SetRequestedRegion(inputPtr->GetLargestPossibleRegion());
  }


template<typename TImage, typename TForegroundConnectivity>
void 
SkeletonizeImageFilter<TImage, TForegroundConnectivity>
::GenerateData()
  {
  this->AllocateOutputs();
  
  typename OrderingImageType::Pointer orderingImage = this->GetOrderingImage();
  
  if(m_SimplicityCriterion.IsNull())
    {
    m_SimplicityCriterion = 
      SimplicityByTopologicalNumbersImageFunction<OutputImageType, 
        TForegroundConnectivity>::New();
    m_SimplicityCriterion->SetInputImage(this->GetInput());
    }
  
  if(m_TerminalityCriterion.IsNull())
    {
    m_TerminalityCriterion = 
      LineTerminalityImageFunction<OutputImageType, 
        TForegroundConnectivity>::New();
    m_TerminalityCriterion->SetInputImage(this->GetInput());
    }
  
  typename OutputImageType::Pointer outputImage = this->GetOutput(0);
  
  // Initialize hierarchical queue
  HierarchicalQueue<typename OrderingImageType::PixelType, 
                     typename InputImageType::IndexType, 
                     std::less<typename OrderingImageType::PixelType> > q;
  
  bool* inQueue = 
    new bool[outputImage->GetRequestedRegion().GetNumberOfPixels()];
  
  // set up progress reporter. There is 2 steps, but we can't know how many 
  // pixels will be in the second one, so use the maximum
  ProgressReporter 
    progress(this, 0, outputImage->GetRequestedRegion().GetNumberOfPixels()*2);
  for(ImageRegionConstIteratorWithIndex<OrderingImageType> 
        it(orderingImage, orderingImage->GetRequestedRegion());
      !it.IsAtEnd(); ++it)
    {
    if(it.Get() != NumericTraits<typename OrderingImageType::PixelType>::Zero )
      {
      q.Push(it.Get(), it.GetIndex());
      inQueue[ outputImage->ComputeOffset(it.GetIndex()) ] = true;
    }
    else
      {
      inQueue[ outputImage->ComputeOffset(it.GetIndex()) ] = false;
      }
    progress.CompletedPixel();
    }
  
  ForegroundConnectivity const & connectivity = 
    ForegroundConnectivity::GetInstance();
  
  while(!q.IsEmpty())
    {
    typename InputImageType::IndexType const current = q.GetFront();
    q.Pop();
    inQueue[outputImage->ComputeOffset(current)] = false;
    
    bool const terminal = m_TerminalityCriterion->EvaluateAtIndex(current);
    bool const simple = m_SimplicityCriterion->EvaluateAtIndex(current);
    
    if( simple && !terminal )
      {
      outputImage->SetPixel(current, 
        NumericTraits<typename OutputImageType::PixelType>::Zero);
      
      // Add neighbors that are not already in the queue
      for(unsigned int i = 0; i < connectivity.GetNumberOfNeighbors(); ++i)
        {
        typename InputImageType::IndexType currentNeighbor;
        for(unsigned int j = 0; j < ForegroundConnectivity::Dimension; ++j)
          {
          currentNeighbor[j] = current[j] + 
            connectivity.GetNeighborsPoints()[i][j];
          }
        
        if( /* currentNeighbor is in image */
              outputImage->GetPixel(currentNeighbor) != 
              NumericTraits<typename OutputImageType::PixelType>::Zero && 
            /* and not in queue */
              !inQueue[outputImage->ComputeOffset(currentNeighbor)]   &&
            /*and has not 0 priority*/
              orderingImage->GetPixel(currentNeighbor) != 
              NumericTraits<typename OrderingImageType::PixelType>::Zero )
          {
          q.Push(orderingImage->GetPixel(currentNeighbor), currentNeighbor);
          inQueue[outputImage->ComputeOffset(currentNeighbor)] = true;
          }
        }
      }
    progress.CompletedPixel();
    }
  
  delete[] inQueue;
}

} // namespace itk

#endif // itkSkeletonizationImageFilter_txx
