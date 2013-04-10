/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkCorrectGMUsingNeighbourhoodFilter_h
#define __itkCorrectGMUsingNeighbourhoodFilter_h

#include "itkImage.h"
#include "itkBaseCTESegmentationFilter.h"

namespace itk
{
/**
 * \class CorrectGMUsingNeighbourhoodFilter
 * \brief Implements section 2.3 in Acosta et al. doi:10.1016/j.media.2009.07.003
 * 
 * Usage:
 * 
 * filter->SetSegmentedImage(segmentedImage);
 * 
 * filter->Update();
 * 
 * The filter takes an image with 3 labels, (GM, WM, CSF). To quote paper,
 * "We implemented an algorithm that checks whether in the 3 x 3 x 3 neighbourhood
 * of each WM boundary voxel, there is any CSF voxel breaking the GM/WM continuity,
 * in which case it is reclassified as GM.".
 * 
 * This exposes 2 lists of indexes, one for all GM pixels before the update, and one
 * for all GM pixels after the update.
 * 
 * \sa niftkCTEBourgeat2008
 *  
 */
template <class TImageType>
class ITK_EXPORT CorrectGMUsingNeighbourhoodFilter : 
    public BaseCTESegmentationFilter< TImageType >
{
public:
  /** Standard "Self" & Superclass typedef.   */
  typedef CorrectGMUsingNeighbourhoodFilter          Self;
  typedef BaseCTESegmentationFilter<TImageType>      Superclass;
  typedef SmartPointer<Self>                         Pointer;
  typedef SmartPointer<const Self>                   ConstPointer;

  /** Method for creation through the object factory.  */
  itkNewMacro(Self);

  /** Run-time type information (and related methods)  */
  itkTypeMacro(CorrectGMUsingNeighbourhoodFilter, BaseCTESegmentationFilter);
  
  /** Image typedef support. */
  typedef typename Superclass::InputImageType          InputImageType;
  typedef typename Superclass::InputPixelType          InputPixelType;
  typedef typename Superclass::InputIndexType          InputIndexType;
  typedef typename Superclass::InputImagePointer       InputImagePointer;
  typedef typename Superclass::InputImageConstPointer  InputImageConstPointer;
  typedef typename Superclass::OutputPixelType         OutputPixelType;
  typedef typename Superclass::OutputImageType         OutputImageType;
  typedef typename OutputImageType::RegionType         OutputImageRegionType;
  typedef typename OutputImageType::IndexType          OutputImageIndexType;
  typedef typename OutputImageType::SizeType           OutputImageSizeType;
  typedef typename Superclass::OutputImagePointer      OutputImagePointer;
  typedef typename Superclass::OutputImageConstPointer OutputImageConstPointer;

  /** Sets the segmented image, at input 0. */
  void SetSegmentedImage(const InputImageType *image) {this->SetNthInput(0, const_cast<InputImageType *>(image)); }

  /** Once the filter has run, a list of GM voxels in original segmented volume */
  const std::vector<InputIndexType>& GetListOfGreyMatterPixelsBeforeCorrection() const;
  
  /** Once the filter has run, a list of GM voxles in corrected segmented volume. */
  const std::vector<InputIndexType>& GetListOfGreyMatterPixelsAfterCorrection() const;
  
  /** So we can extract the number reclassified immediately after an update. */
  itkGetMacro(NumberReclassified, unsigned long int);
  
  /** Set/Get flag to determine if we examine the full 27 connected, or just the local 6 connected neighbourhood. Default true. */
  itkSetMacro(UseFullNeighbourHood, bool);
  itkGetMacro(UseFullNeighbourHood, bool);
  
protected:
  
  CorrectGMUsingNeighbourhoodFilter();
  virtual ~CorrectGMUsingNeighbourhoodFilter()  {};

  /** Standard Print Self. */
  virtual void PrintSelf(std::ostream&, Indent) const;

  /* The main filter method. Note, single threaded. */
  virtual void GenerateData();
  
private:

  CorrectGMUsingNeighbourhoodFilter(const Self&); // purposely not implemented
  void operator=(const Self&); // purposely not implemented

  std::vector<InputIndexType> m_ListOfGreyMatterVoxelsBeforeCorrection;
  
  std::vector<InputIndexType> m_ListOfGreyMatterVoxelsAfterCorrection;
  
  bool m_UseFullNeighbourHood;
  
  unsigned long int m_NumberReclassified;
};
  
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkCorrectGMUsingNeighbourhoodFilter.txx"
#endif

#endif
