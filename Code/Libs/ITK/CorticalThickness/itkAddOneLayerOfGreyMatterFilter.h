/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkAddOneLayerOfGreyMatterFilter_h
#define __itkAddOneLayerOfGreyMatterFilter_h

#include <itkImage.h>
#include "itkBaseCTESegmentationFilter.h"


namespace itk
{
/**
 * \class AddOneLayerOfGreyMatterFilter
 * \brief Assumes input is GM, WM, CSF labelled image, and adds 1 layer of GM to the WM,
 * which means that the output will have the same size WM, only 1 voxel wide layer of GM,
 * and the rest is then all CSF.
 * 
 * This is inspired by Hutton et. al. NeuroImage 2008, doi:10.1016/j.neuroimage.2008.01.027,
 * in the section of 'Preserving cortical topography'. 
 */
template <class TImageType>
class ITK_EXPORT AddOneLayerOfGreyMatterFilter : 
    public BaseCTESegmentationFilter< TImageType >
{
public:
  /** Standard "Self" & Superclass typedef.   */
  typedef AddOneLayerOfGreyMatterFilter              Self;
  typedef BaseCTESegmentationFilter<TImageType>      Superclass;
  typedef SmartPointer<Self>                         Pointer;
  typedef SmartPointer<const Self>                   ConstPointer;

  /** Method for creation through the object factory.  */
  itkNewMacro(Self);

  /** Run-time type information (and related methods)  */
  itkTypeMacro(AddOneLayerOfGreyMatterFilter, BaseCTESegmentationFilter);
  
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
  
  /** Get the number of grey voxels in boundary layer */
  itkGetMacro(NumberOfGreyInBoundaryLayer, unsigned long int);

  /** Get the number of grey voxels left outside boundary layer. */
  itkGetMacro(NumberOfGreyLeftOutsideBoundaryLayer, unsigned long int);

protected:
  
  AddOneLayerOfGreyMatterFilter();
  virtual ~AddOneLayerOfGreyMatterFilter()  {};

  /** Standard Print Self. */
  virtual void PrintSelf(std::ostream&, Indent) const;

  /* The main filter method. Note, single threaded. */
  virtual void GenerateData();
  
private:

  AddOneLayerOfGreyMatterFilter(const Self&); // purposely not implemented
  void operator=(const Self&); // purposely not implemented

  unsigned long int m_NumberOfGreyInBoundaryLayer;
  
  unsigned long int m_NumberOfGreyLeftOutsideBoundaryLayer;
  
};
  
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkAddOneLayerOfGreyMatterFilter.txx"
#endif

#endif
