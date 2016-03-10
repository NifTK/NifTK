/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "itkMIDASRegionOfInterestCalculator.h"
#include <itkConversionUtils.h>
#include <itkSpatialOrientationAdapter.h>
#include <itkImageRegionConstIteratorWithIndex.h>

namespace itk
{

template<class TPixel, unsigned int VImageDimension>
MIDASRegionOfInterestCalculator<TPixel, VImageDimension>
::MIDASRegionOfInterestCalculator()
{
}

template<class TPixel, unsigned int VImageDimension>
std::string 
MIDASRegionOfInterestCalculator<TPixel, VImageDimension>
::GetOrientationString(ImageType* image)
{
  DirectionType direction = image->GetDirection();
  
  typename itk::SpatialOrientationAdapter adaptor;
  typename itk::SpatialOrientation::ValidCoordinateOrientationFlags orientationFlag;
  orientationFlag = adaptor.FromDirectionCosines(direction);
  std::string orientationString = itk::ConvertSpatialOrientationToString(orientationFlag);

  itkDebugMacro(<< "::GetOrientationString direction=\n" << direction << ", which has orientationString=" << orientationString);

  if (orientationString == "UNKNOWN")
  {
    itkExceptionMacro(<< "Couldn't work out orientationString for direction=\n" << direction);
  }
  
  return orientationString;  
}

template<class TPixel, unsigned int VImageDimension>
int
MIDASRegionOfInterestCalculator<TPixel, VImageDimension>
::GetAxis(ImageType* image, Orientation orientation)
{
  std::string orientationString = this->GetOrientationString(image);

  int outputAxis = -1;
  
  if (orientationString != "ORIENTATION_UNKNOWN")
  {
    for (unsigned int i = 0; i < VImageDimension; i++)
    {
      if (orientation == ORIENTATION_AXIAL && (orientationString[i] == 'S' || orientationString[i] == 'I'))
      {
        outputAxis = i;
        break;
      }

      if (orientation == ORIENTATION_CORONAL && (orientationString[i] == 'A' || orientationString[i] == 'P'))
      {
        outputAxis = i;
        break;
      }

      if (orientation == ORIENTATION_SAGITTAL && (orientationString[i] == 'L' || orientationString[i] == 'R'))
      {
        outputAxis = i;
        break;
      }
    }
  }
  
  itkDebugMacro(<< "::GetAxis orientationString=" << orientationString << ", looking for axis=" << orientation << ", returns axis=" << outputAxis);
  
  if (outputAxis == -1)
  {
    itkExceptionMacro(<< "Couldn't work out axis for orientationString=" << orientationString << ", and orientation=" << orientation);
  }
  
  return outputAxis;
}

template<class TPixel, unsigned int VImageDimension>
int 
MIDASRegionOfInterestCalculator<TPixel, VImageDimension>
::GetPlusOrUpDirection(ImageType* image, Orientation orientation)
{
  std::string orientationString = this->GetOrientationString(image);
  int axis = this->GetAxis(image, orientation);
  
  int direction = 0;
  
  if (axis != -1 && axis < (int)VImageDimension && orientationString != "UNKNOWN")
  {
    // Note, in Nifti speak, when a volume is LPS we have:
    // x axis = Left to Right
    // y axis = Posterior to Anterior
    // z axis = Superior to Inferior
    // But in ITK, ITK would call this the exact opposite, i.e. RAI.
    // So, the ITK orientationString tells you where each axis starts from.
    
    if (orientationString[axis] == 'L'
     || orientationString[axis] == 'P'
     || orientationString[axis] == 'S'
     )
    {
      direction = 1;
    }
    else if (orientationString[axis] == 'R'
     || orientationString[axis] == 'A'
     || orientationString[axis] == 'I'
     )
    {
      direction = -1;
    }
  }
  
  itkDebugMacro(<< "::GetPlusOrUpDirection For orientationString=" << orientationString << ", axis=" << axis << ", plus direction=" << direction);
  
  if (direction == 0)
  {
    itkExceptionMacro(<< "orientationString=" << orientationString << ", and axis=" << axis << ", and can't work out direction");
  }
  return direction;
}

template<class TPixel, unsigned int VImageDimension>
void 
MIDASRegionOfInterestCalculator<TPixel, VImageDimension>
::CheckSliceNumber(ImageType* image, Orientation orientation, int sliceNumber)
{
  assert(image);

  RegionType region = image->GetLargestPossibleRegion();
  SizeType size = region.GetSize();

  // Check slice number is within region.
  
  int axis = this->GetAxis(image, orientation);      
  if (axis < 0 || axis >= (int)VImageDimension)
  {
    itkExceptionMacro(<< "Invalid axis number:" << axis);
  }
  
  if (sliceNumber < 0 || sliceNumber >= (int)size[axis])
  {
    itkExceptionMacro(<< "Invalid slice number:" << sliceNumber);
  }
}

template<class TPixel, unsigned int VImageDimension>
typename MIDASRegionOfInterestCalculator<TPixel, VImageDimension>::RegionType
MIDASRegionOfInterestCalculator<TPixel, VImageDimension> 
::GetRegion(typename MIDASRegionOfInterestCalculator<TPixel, VImageDimension>::ImageType* image, 
  Orientation orientation, 
  int sliceNumber,
  bool doSingleSlice,
  bool doPlus
  )
{
  assert(image);
  this->CheckSliceNumber(image, orientation, sliceNumber);
  
  std::string orientationStringForDebug = this->GetOrientationString(image);
  int axis = this->GetAxis(image, orientation);
  
  int direction = this->GetPlusOrUpDirection(image, orientation);
  if (!doPlus)
  {
    direction *= -1;
  }
   
  RegionType region = image->GetLargestPossibleRegion();
  IndexType outputIndex = region.GetIndex();
  SizeType outputSize = region.GetSize();

  if (doSingleSlice)
  {
    outputIndex[axis] = sliceNumber;
    outputSize[axis] = 1;
  }
  else
  {
    if (direction == -1)
    {
      outputIndex[axis] = 0;
      outputSize[axis] = sliceNumber;
    }
    else
    {
      outputIndex[axis] = sliceNumber + 1;
      outputSize[axis] = outputSize[axis] - sliceNumber - 1;
    }
  }  
  itkDebugMacro(<< "::GetRegion With largest possible region=\n" << region << ", and orientationString=" << orientationStringForDebug << ", doPlus=" << doPlus << ", sliceNumber=" << sliceNumber << ", direction=" << direction << ", axis=" << axis << ", outputSize=" << outputSize << ", outputIndex=" << outputIndex);

  region.SetSize(outputSize);
  region.SetIndex(outputIndex);
  return region;  
}

template<class TPixel, unsigned int VImageDimension>
std::vector< typename itk::Image<TPixel, VImageDimension>::RegionType >
MIDASRegionOfInterestCalculator<TPixel, VImageDimension> 
::GetRegionAsSlices(typename itk::Image<TPixel, VImageDimension>* image, 
  Orientation orientation, 
  int sliceNumber,
  bool doSingleSlice,
  bool doPlus
  )
{
  std::vector<RegionType> results;
  
  assert(image);
  this->CheckSliceNumber(image, orientation, sliceNumber);
  
  std::string orientationStringForDebug = this->GetOrientationString(image);
  int axis = this->GetAxis(image, orientation);
  
  int direction = this->GetPlusOrUpDirection(image, orientation);
  if (!doPlus)
  {
    direction *= -1;
  }

  RegionType region = image->GetLargestPossibleRegion();
  IndexType largestIndex = region.GetIndex();
  SizeType largestSize = region.GetSize();
   
  int startSlice = 0;
  int endSlice = 0;
  
  if (doSingleSlice)
  {
    startSlice = sliceNumber;
    endSlice = sliceNumber;
  }
  else
  {
    if (direction == -1)
    {
      startSlice = sliceNumber -1;
      endSlice = 0;
    }
    else
    {
      startSlice = sliceNumber +1;
      endSlice = largestSize[axis] -1;
    }
  }
  
  itkDebugMacro(<< "::GetRegionAsSlices With largest possible region=\n" << region << ", and orientationString=" << orientationStringForDebug << ", doPlus=" << doPlus << ", sliceNumber=" << sliceNumber << ", direction=" << direction << ", axis=" << axis << ", startSlice=" << startSlice << ", endSlice=" << endSlice);

  if (startSlice < 0 || startSlice == (int)largestSize[axis])
  {
    itkDebugMacro(<< "::GetRegionAsSlices startSlice=" << startSlice << ", endSlice=" << endSlice << ", so nothing to do");
    return results;
  }
  
  int numberOfExpectedSlices = abs(endSlice - startSlice) + 1;
  int currentSlice = startSlice;
  for (int i = 0; i < numberOfExpectedSlices; i++)
  {
    RegionType outputRegion;
    IndexType  outputIndex;
    SizeType   outputSize;
    
    outputIndex = largestIndex;
    outputSize = largestSize;
    
    outputSize[axis] = 1;
    outputIndex[axis] = currentSlice;
    
    outputRegion.SetSize(outputSize);
    outputRegion.SetIndex(outputIndex);
    
    results.push_back(outputRegion);
    currentSlice += direction;
  }
  
  return results;
}

template<class TPixel, unsigned int VImageDimension>
std::vector< typename itk::Image<TPixel, VImageDimension>::RegionType >
MIDASRegionOfInterestCalculator<TPixel, VImageDimension> 
::SplitRegionBySlices(
  typename itk::Image<TPixel, VImageDimension>::RegionType region,
  typename itk::Image<TPixel, VImageDimension>* image, 
  Orientation orientation
  )
{
  std::vector<RegionType> results;
  
  assert(image);
  if (!image->GetLargestPossibleRegion().IsInside(region))
  {
    itkExceptionMacro(<< "Supplied region=\n" << region << ", is not inside the supplied image region=\n" << image->GetLargestPossibleRegion());
  }
  
  int axis = this->GetAxis(image, orientation);
  int direction = this->GetPlusOrUpDirection(image, orientation);
  IndexType regionIndex = region.GetIndex();
  SizeType regionSize = region.GetSize();
  
  int numberOfExpectedSlices = regionSize[axis];
  int startSlice = 0;
  
  if (direction == 1)
  {
    startSlice = regionIndex[axis];
  }
  else
  {
    startSlice = regionIndex[axis] + regionSize[axis] -1;
  }
  
  int currentSlice = startSlice;
  for (int i = 0; i < numberOfExpectedSlices; i++)
  {
    RegionType outputRegion;
    IndexType  outputIndex;
    SizeType   outputSize;
    
    outputIndex = regionIndex;
    outputSize = regionSize;
    
    outputSize[axis] = 1;
    outputIndex[axis] = currentSlice;
    
    outputRegion.SetSize(outputSize);
    outputRegion.SetIndex(outputIndex);
    
    results.push_back(outputRegion);
    currentSlice += direction;
  }
 
  return results; 
}

template<class TPixel, unsigned int VImageDimension>
typename MIDASRegionOfInterestCalculator<TPixel, VImageDimension>::RegionType
MIDASRegionOfInterestCalculator<TPixel, VImageDimension> 
::GetPlusOrUpRegion(typename MIDASRegionOfInterestCalculator<TPixel, VImageDimension>::ImageType* image, 
  Orientation orientation, 
  int sliceNumber
  )
{
  return this->GetRegion(image, orientation, sliceNumber, false, true);
}

template<class TPixel, unsigned int VImageDimension>
typename MIDASRegionOfInterestCalculator<TPixel, VImageDimension>::RegionType
MIDASRegionOfInterestCalculator<TPixel, VImageDimension>
::GetMinusOrDownRegion(typename MIDASRegionOfInterestCalculator<TPixel, VImageDimension>::ImageType* image, 
  Orientation orientation, 
  int sliceNumber
  )
{
  return this->GetRegion(image, orientation, sliceNumber, false, false);
}

template<class TPixel, unsigned int VImageDimension>
typename MIDASRegionOfInterestCalculator<TPixel, VImageDimension>::RegionType
MIDASRegionOfInterestCalculator<TPixel, VImageDimension>
::GetSliceRegion(typename MIDASRegionOfInterestCalculator<TPixel, VImageDimension>::ImageType* image, 
  Orientation orientation, 
  int sliceNumber
  )
{
  return this->GetRegion(image, orientation, sliceNumber, true, false);
}

template<class TPixel, unsigned int VImageDimension>
std::vector< typename itk::Image<TPixel, VImageDimension>::RegionType >
MIDASRegionOfInterestCalculator<TPixel, VImageDimension> 
::GetPlusOrUpRegionAsSlices(typename itk::Image<TPixel, VImageDimension>* image, 
  Orientation orientation, 
  int sliceNumber
  )
{
  return GetRegionAsSlices(image, orientation, sliceNumber, false, true);
}

template<class TPixel, unsigned int VImageDimension>
std::vector< typename itk::Image<TPixel, VImageDimension>::RegionType >
MIDASRegionOfInterestCalculator<TPixel, VImageDimension> 
::GetMinusOrDownRegionAsSlices(typename itk::Image<TPixel, VImageDimension>* image, 
  Orientation orientation, 
  int sliceNumber
  )
{
  return GetRegionAsSlices(image, orientation, sliceNumber, false, false);
}

template<class TPixel, unsigned int VImageDimension>
typename itk::Image<TPixel, VImageDimension>::RegionType
MIDASRegionOfInterestCalculator<TPixel, VImageDimension>
::GetMinimumRegion(ImageType *image, PixelType background)
{
  IndexType min;
  IndexType max;
  IndexType current;
  
  min.Fill(std::numeric_limits<typename IndexType::IndexValueType>::max());
  max.Fill(std::numeric_limits<typename IndexType::IndexValueType>::min());
  
  ImageRegionConstIteratorWithIndex<ImageType> iterator(image, image->GetLargestPossibleRegion());
  for (iterator.GoToBegin(); !iterator.IsAtEnd(); ++iterator)
  {
    if (iterator.Get() != background)
    {
      current = iterator.GetIndex();
      for (unsigned int i = 0; i < VImageDimension; i++)
      {
        if (current[i] < min[i])
        {
          min[i] = current[i];
        }
        else if (current[i] > max[i])
        {
          max[i] = current[i];
        }
      }
    } 
  }
  
  SizeType outputSize;
  IndexType outputIndex;
  RegionType outputRegion;
  
  for (unsigned int i = 0; i < VImageDimension; i++)
  {
    outputSize[i] = max[i] - min[i] + 1;
    outputIndex[i] = min[i];
  }
  
  outputRegion.SetSize(outputSize);
  outputRegion.SetIndex(outputIndex);
  
  itkDebugMacro( << "::GetMinimumRegion return\n" << outputRegion);
  
  return outputRegion;
}

} // end namespace
