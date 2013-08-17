/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkMIDASRegionOfInterestCalculator_h
#define itkMIDASRegionOfInterestCalculator_h

#include <itkObject.h>
#include <itkObjectFactory.h>
#include <itkImage.h>
#include <itkMIDASHelper.h>

namespace itk
{

/**
 * \class MIDASRegionOfInterestCalculator.
 * \brief Class to calculate regions within an image according to MIDAS specifications,
 * where for example we need to know (for Wipe+, Wipe- and PropUp and PropDown) that
 * "plus" or "up" means anterior, right or superior and "minus" or "down" means
 * posterior, left or inferior.
 * \deprecated See MIDASGeneralSegmentorView now uses itk::ImageUpdateCopyRegionProcessor.
 */
template <class TPixel, unsigned int VImageDimension>
class ITK_EXPORT MIDASRegionOfInterestCalculator : public Object {

public:

  /** Standard class typedefs */
  typedef MIDASRegionOfInterestCalculator Self;
  typedef Object                          Superclass;
  typedef SmartPointer<Self>              Pointer;
  typedef SmartPointer<const Self>        ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(MIDASRegionOfInterestCalculator, Object);

  /** Dimension of the image.  This constant is used by functions that are
   * templated over image type (as opposed to being templated over pixel type
   * and dimension) when they need compile time access to the dimension of
   * the image. */
  itkStaticConstMacro(ImageDimension, unsigned int, VImageDimension);

  /** Additional typedefs */
  typedef TPixel PixelType;
  typedef Image<TPixel, VImageDimension>    ImageType;
  typedef typename ImageType::Pointer       ImagePointer;
  typedef typename ImageType::IndexType     IndexType;
  typedef typename ImageType::SizeType      SizeType;
  typedef typename ImageType::RegionType    RegionType;
  typedef typename ImageType::DirectionType DirectionType;

  /** Retrieves the orientation string such as RPI, LAS etc. */
  std::string GetOrientationString(ImageType* image);

  /** Retrieves the index of the image axis corresponding to the currentOrientation, in 3D will return 0,1,2 or -1 if not found. */
  int GetAxis(ImageType* image, ORIENTATION_ENUM currentOrientation);

  /** Retrieves the direction that corresponds to "Plus" or "Up", which is towards anterior, right or superior and will return +1 for increasing, -1 for decreasing or 0 if not found. */
  int GetPlusOrUpDirection(ImageType* image, ORIENTATION_ENUM currentOrientation);

  /** From the given image, orientation and slice number will calculate the Plus/Up region. */
  RegionType GetPlusOrUpRegion(ImageType* image, ORIENTATION_ENUM currentOrientation, int sliceNumber);

  /** From the given image, orientation and slice number will calculate the Minus/Down region. */
  RegionType GetMinusOrDownRegion(ImageType* image, ORIENTATION_ENUM currentOrientation, int sliceNumber);

  /** From the given image, orientation and slice number, will calculate the region for the current slice. */
  RegionType GetSliceRegion(ImageType* image, ORIENTATION_ENUM currentOrientation, int sliceNumber);

  /** From the give image, orientation and slice number will calculate the Plus/Up region and return a list of regions corresponding to each slice, so you can iterate slice-wise. */
  std::vector<RegionType> GetPlusOrUpRegionAsSlices(ImageType* image, ORIENTATION_ENUM currentOrientation, int sliceNumber);

  /** From the give image, orientation and slice number will calculate the Minus/Down region and return a list of regions corresponding to each slice, so you can iterate slice-wise. */
  std::vector<RegionType> GetMinusOrDownRegionAsSlices(ImageType* image, ORIENTATION_ENUM currentOrientation, int sliceNumber);

  /** Takes a given region of interest, and splits it down into slices according to orientation. */
  std::vector<RegionType> SplitRegionBySlices(RegionType regionOfInterest, ImageType* image, ORIENTATION_ENUM currentOrientation);

  /** Works out the minimum region to encapsulate data that does not contain the background value. */
  RegionType GetMinimumRegion(ImageType *image, PixelType background);

protected:
  MIDASRegionOfInterestCalculator();
  virtual ~MIDASRegionOfInterestCalculator() {}

private:
  MIDASRegionOfInterestCalculator(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  /** Will throw itkException if slice number is invalid (outside region). */
  void CheckSliceNumber(ImageType* image, ORIENTATION_ENUM currentOrientation, int sliceNumber);

  /** Calculates the region, either side of the sliceNumber, given the currentOrientation. */
  RegionType GetRegion(ImageType* image, ORIENTATION_ENUM currentOrientation, int sliceNumber, bool doSingleSlice, bool doPlus);

  /** Calculates the region, either side of the sliceNumber, given the currentOrientation. returning each slice to process as a separate region of interest. */
  std::vector<RegionType> GetRegionAsSlices(ImageType* image, ORIENTATION_ENUM currentOrientation, int sliceNumber, bool doSingleSlice, bool doPlus);

}; // end class

} // end namespace

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMIDASRegionOfInterestCalculator.txx"
#endif

#endif // ITKMIDASREGIONOFINTERESTCALCULATOR_H
