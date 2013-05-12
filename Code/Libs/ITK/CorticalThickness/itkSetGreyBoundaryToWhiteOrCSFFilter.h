/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkSetGreyBoundaryToWhiteOrCSFFilter_h
#define __itkSetGreyBoundaryToWhiteOrCSFFilter_h

#include <itkImage.h>
#include "itkBaseCTESegmentationFilter.h"

namespace itk
{
/**
 * \class SetGreyBoundaryToWhiteOrCSFFilter
 * \brief Assumes input is GM, WM, CSF labelled image, where the GM is exaclty 1 
 * voxel wide, we then set this GM voxel to WM or CSF depending on thickness,
 * and the immediate neighbourhood.
 * 
 * This is to implement Hutton et. al. NeuroImage 2008, doi:10.1016/j.neuroimage.2008.01.027,
 * in the section of 'Preserving cortical topography'. 
 */
template <class TImageType, typename TScalarType, unsigned int NDimensions>
class ITK_EXPORT SetGreyBoundaryToWhiteOrCSFFilter : 
    public BaseCTESegmentationFilter< TImageType >
{
public:
  /** Standard "Self" & Superclass typedef.   */
  typedef SetGreyBoundaryToWhiteOrCSFFilter          Self;
  typedef BaseCTESegmentationFilter<TImageType>      Superclass;
  typedef SmartPointer<Self>                         Pointer;
  typedef SmartPointer<const Self>                   ConstPointer;

  /** Method for creation through the object factory.  */
  itkNewMacro(Self);

  /** Run-time type information (and related methods)  */
  itkTypeMacro(SetGreyBoundaryToWhiteOrCSFFilter, BaseCTESegmentationFilter);

  /** Get the number of dimensions we are working in. */
  itkStaticConstMacro(Dimension, unsigned int, NDimensions);

  /** Image typedef support. */
  typedef typename Superclass::InputImageType             InputImageType;
  typedef typename Superclass::InputPixelType             InputPixelType;
  typedef typename Superclass::InputIndexType             InputIndexType;
  typedef typename Superclass::InputImagePointer          InputImagePointer;
  typedef typename Superclass::InputImageConstPointer     InputImageConstPointer;
  typedef typename Superclass::OutputPixelType            OutputPixelType;
  typedef typename Superclass::OutputImageType            OutputImageType;
  typedef typename Superclass::OutputImagePointer         OutputImagePointer;
  typedef typename Superclass::OutputImageConstPointer    OutputImageConstPointer;
  typedef TScalarType                                     ThicknessPixelType;
  typedef Image<ThicknessPixelType, NDimensions>          ThicknessImageType;

  /** Sets the label image at input 0. */
  void SetLabelImage(const InputImageType *image) {this->SetNthInput(0, const_cast<InputImageType *>(image)); }

  /** Sets the one layer image at input 1. */
  void SetOneLayerImage(const InputImageType *image) {this->SetNthInput(1, const_cast<InputImageType *>(image)); }

  /** Sets the thickness image at input 2. */
  void SetThicknessImage(const ThicknessImageType *image) {this->SetNthInput(2, const_cast<ThicknessImageType *>(image)); }

  /** Set/Get the expected voxel size. */
  itkSetMacro(ExpectedVoxelSize, float);
  itkGetMacro(ExpectedVoxelSize, float);
  
  /** Set/Get the label that we tag the changed CSF with. Default 4. */
  itkSetMacro(TaggedCSFLabel, InputPixelType);
  itkGetMacro(TaggedCSFLabel, InputPixelType);
  
  /** Get the number of Grey left before a successful update. */
  itkGetMacro(NumberOfGreyBefore, unsigned long int);
  
  /** Get the number of Grey left after a successful update. */
  itkGetMacro(NumberOfGreyAfter, unsigned long int);
  
protected:
  
  SetGreyBoundaryToWhiteOrCSFFilter();
  virtual ~SetGreyBoundaryToWhiteOrCSFFilter()  {};

  /** Standard Print Self. */
  virtual void PrintSelf(std::ostream&, Indent) const;

  /* The main filter method. Note, single threaded. */
  virtual void GenerateData();
  
private:

  SetGreyBoundaryToWhiteOrCSFFilter(const Self&); // purposely not implemented
  void operator=(const Self&); // purposely not implemented
  
  float m_ExpectedVoxelSize;
  
  InputPixelType m_TaggedCSFLabel;
  
  unsigned long int m_NumberOfGreyBefore;
  
  unsigned long int m_NumberOfGreyAfter;
  
};
  
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkSetGreyBoundaryToWhiteOrCSFFilter.txx"
#endif

#endif
