/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkCorrectGMUsingPVMapFilter_h
#define __itkCorrectGMUsingPVMapFilter_h

#include "itkImage.h"
#include "itkBaseCTESegmentationFilter.h"


namespace itk
{
/**
 * \class CorrectGMUsingPVMapFilter
 * \brief Implements section 2.3.1 Correction Of Segmentation in Bourgeat MICCAI 2008.
 * 
 * This class provides the segmentation correction as described in Bourgeat MICCAI 2008.
 * 
 * Usage:
 * 
 * filter->SetSegmentedImage(segmentedImage);
 * 
 * filter->SetGMPVMap(greyMatterPVMap);
 * 
 * filter->SetLabelThresholds(grey, white, csf); 
 * 
 * filter->Update();
 * 
 * The filter implements two points. Briefly, if a GM voxel is on the GM/CSF boundary
 * it is re-classified as CSF if GMPVC < 1. Also, if a CSF (WM resp) is on CSF/WM boundary
 * its reclassified as GM regardless of the GMPVC fractional content.
 * You should set the GM, WM and CSF boundaries yourself before calling Update, as its quicker.
 * 
 * This exposes 2 lists of indexes, one for all GM pixels before the update, and one
 * for all GM pixels after the update.
 * 
 * \sa niftkCTEBourgeat2008
 *  
 */
template <class TImageType>
class ITK_EXPORT CorrectGMUsingPVMapFilter : 
    public BaseCTESegmentationFilter< TImageType >
{
public:
  /** Standard "Self" & Superclass typedef.   */
  typedef CorrectGMUsingPVMapFilter                  Self;
  typedef BaseCTESegmentationFilter<TImageType>      Superclass;
  typedef SmartPointer<Self>                         Pointer;
  typedef SmartPointer<const Self>                   ConstPointer;

  /** Method for creation through the object factory.  */
  itkNewMacro(Self);

  /** Run-time type information (and related methods)  */
  itkTypeMacro(CorrectGMUsingPVMap, BaseCTESegmentationFilter);
  
  /** Image typedef support. */
  typedef typename Superclass::InputImageType          InputImageType;
  typedef typename Superclass::InputPixelType          InputPixelType;
  typedef typename Superclass::InputIndexType          InputIndexType;
  typedef typename Superclass::InputImagePointer       InputImagePointer;
  typedef typename Superclass::InputImageConstPointer  InputImageConstPointer;
  typedef typename Superclass::OutputPixelType         OutputPixelType;
  typedef typename Superclass::OutputImageType         OutputImageType;
  typedef typename Superclass::OutputImagePointer      OutputImagePointer;
  typedef typename Superclass::OutputImageConstPointer OutputImageConstPointer;

  /** Sets the segmented image, at input 0. */
  void SetSegmentedImage(const InputImageType *image) {this->SetNthInput(0, const_cast<InputImageType *>(image)); }

  /** Sets the pv map, at input 1. */
  void SetGMPVMap(const InputImageType *image) {this->SetNthInput(1, const_cast<InputImageType *>(image)); }
  
  /** Once the filter has run, a list of GM voxels in original segmented volume */
  const std::vector<InputIndexType>& GetListOfGreyMatterPixelsBeforeCorrection() const;
  
  /** Once the filter has run, a list of GM voxles in corrected segmented volume. */
  const std::vector<InputIndexType>& GetListOfGreyMatterPixelsAfterCorrection() const;
  
  /** Set the grey matter threshold. Defaults to 1. */
  itkSetMacro(GreyMatterThreshold, double);
  itkGetMacro(GreyMatterThreshold, double);

  /** If true, we do the grey matter check, if false we don't. Default true. */
  itkSetMacro(DoGreyMatterCheck, bool);
  itkGetMacro(DoGreyMatterCheck, bool);

  /** If true, we do the CSF check, if false we don't. Default true. */
  itkSetMacro(DoCSFCheck, bool);
  itkGetMacro(DoCSFCheck, bool);

protected:
  
  CorrectGMUsingPVMapFilter();
  virtual ~CorrectGMUsingPVMapFilter()  {};

  /** Standard Print Self. */
  virtual void PrintSelf(std::ostream&, Indent) const;

  /* The main filter method. Note, single threaded. */
  virtual void GenerateData();
  
private:

  CorrectGMUsingPVMapFilter(const Self&); // purposely not implemented
  void operator=(const Self&); // purposely not implemented

  std::vector<InputIndexType> m_ListOfGreyMatterVoxelsBeforeCorrection;
  
  std::vector<InputIndexType> m_ListOfGreyMatterVoxelsAfterCorrection;
  
  bool m_DoGreyMatterCheck;
  
  bool m_DoCSFCheck;
  
  double m_GreyMatterThreshold;
  
};
  
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkCorrectGMUsingPVMapFilter.txx"
#endif

#endif
