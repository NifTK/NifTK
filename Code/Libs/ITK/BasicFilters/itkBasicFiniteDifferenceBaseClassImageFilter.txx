/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkBasicFiniteDifferenceBaseClassImageFilter_txx
#define __itkBasicFiniteDifferenceBaseClassImageFilter_txx

#include "itkBasicFiniteDifferenceBaseClassImageFilter.h"

#include "itkLogHelper.h"

namespace itk
{

template <typename TInputImage, typename TOutputImage>
BasicFiniteDifferenceBaseClassImageFilter<TInputImage, TOutputImage>
::BasicFiniteDifferenceBaseClassImageFilter()
{
}

template <typename TInputImage, typename TOutputImage>
BasicFiniteDifferenceBaseClassImageFilter<TInputImage, TOutputImage>
::~BasicFiniteDifferenceBaseClassImageFilter()
{
}

template <typename TInputImage, typename TOutputImage>
void
BasicFiniteDifferenceBaseClassImageFilter<TInputImage, TOutputImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
}

template <typename TInputImage, typename TOutputImage>
void
BasicFiniteDifferenceBaseClassImageFilter<TInputImage, TOutputImage>
::BeforeThreadedGenerateData()
{
  TOutputImage *outputImage = static_cast< TOutputImage * >(this->ProcessObject::GetOutput(0));
  outputImage->FillBuffer(0);
}

template <typename TInputImage, typename TOutputImage>
typename BasicFiniteDifferenceBaseClassImageFilter<TInputImage, TOutputImage>::ImageRegionType 
BasicFiniteDifferenceBaseClassImageFilter<TInputImage, TOutputImage>
::CheckAndAdjustRegion(const ImageRegionType& region, TInputImage* image)
{
  int offset = 2;
  
  SizeType size = image->GetLargestPossibleRegion().GetSize();
  
  ImageRegionType newRegion;
  IndexType newIndex = region.GetIndex();
  SizeType newSize = region.GetSize();
  
  int difference = 0;
  
  for (int i = 0; i < Dimension; i++)
  {
    if (newIndex[i] < offset)
    {
      difference = abs(newIndex[i] - offset);
      newIndex[i] += difference;
      newSize[i] -= difference;
    }
    
    if (newIndex[i] > (int)(size[i] - 1 - offset))
    {
      newIndex[i] = size[i] - 1 - offset;
      newSize[i] = 1;
    }
          
    if (newIndex[i] + newSize[i] > (size[i] - 1 - offset))
    {
      newSize[i] = size[i] - offset - newIndex[i];
    }
  }

  newRegion.SetIndex(newIndex);
  newRegion.SetSize(newSize);
  
  niftkitkDebugMacro(<< "CheckAndAdjustRegion():inputSize=" << region.GetSize() \
    << ", inputIndex=" << region.GetIndex() \
    << ", outputSize=" << newRegion.GetSize() \
    << ", outputIndex=" << newRegion.GetIndex() \
    );
    
  return newRegion;  
}


template <typename TInputImage, typename TOutputImage>
double 
BasicFiniteDifferenceBaseClassImageFilter<TInputImage, TOutputImage>
::d(int dimension, IndexType& location, TInputImage* image)
{

  typedef typename TInputImage::SpacingType				ImageSpacingType;
  ImageSpacingType delta;
  delta = image->GetSpacing();
   
  double result = 0;
  IndexType tmpLocation = location;
  
  tmpLocation[dimension] += 1;
  result += image->GetPixel(tmpLocation);
  
  tmpLocation[dimension] -= 2;
  result -= image->GetPixel(tmpLocation);
  
  result /= (2.0*delta[dimension]);
    
    
  return result;
}

template <typename TInputImage, typename TOutputImage>
double
BasicFiniteDifferenceBaseClassImageFilter<TInputImage, TOutputImage>
::dd(int dimension, IndexType& location, TInputImage* image)
{
 
  typedef typename TInputImage::SpacingType				ImageSpacingType;
  ImageSpacingType delta;
  delta = image->GetSpacing();
 
  double result = 0;
  IndexType tmpLocation = location;
  
  result -= 2 * image->GetPixel(tmpLocation);
  
  tmpLocation[dimension] += 1;
  result += image->GetPixel(tmpLocation);
  
  tmpLocation[dimension] -= 2;
  result += image->GetPixel(tmpLocation);
  
  result /= pow(delta[dimension],2);
 
 // Note that we are assuming that the distance between each sample point is unit length (i.e. 1)

  return result;

}

template <typename TInputImage, typename TOutputImage>
double
BasicFiniteDifferenceBaseClassImageFilter<TInputImage, TOutputImage>
::dd(int dimension1, int dimension2, IndexType& location, TInputImage* image)
{
	if (dimension1 == dimension2)
	{
		double result = d(dimension1, location, image);
		return result;
	}
	
	// The mathematical method here is to find the differential of an image along dimension 2.
	// Then, the result is differentiated again along dimension one.
	// In this implementation, since we are considering only one voxel at a time, the method is as follows:
	// For the central voxel, the two neighbouring voxel positions along dimension one are selected.
	// The differential of those points along dimension 2 is determined.
	// Then these two values are used to give the differential dd(dim1)(dim2) for the central voxel.
	
	
	typedef typename TInputImage::SpacingType				ImageSpacingType;
    ImageSpacingType delta;
    delta = image->GetSpacing();
	
	double df [2] = { 0, 0 };

	IndexType tmpLocation;
 
 	tmpLocation = location;
 	
 	tmpLocation[dimension1] += 1;
 	tmpLocation[dimension2] += 1;
 	df[0] += image->GetPixel(tmpLocation);
 	tmpLocation[dimension2] -= 2;
 	df[0] -= image->GetPixel(tmpLocation);
    df[0] /= 2.0*(delta[dimension2]);

 	 	
 	
 	tmpLocation[dimension1] -= 2;
 	df[1] -= image->GetPixel(tmpLocation);
 	tmpLocation[dimension2] += 2;
 	df[1] += image->GetPixel(tmpLocation);
 	df[1] /= 2.0 * (delta[dimension2]);

 	
 	
 	double result = 0;
  	result += df[0];
	result -= df[1];
	result /= 2.0 * (delta[dimension1]);

	 
	 
  return result;
}

} // end namespace

#endif
