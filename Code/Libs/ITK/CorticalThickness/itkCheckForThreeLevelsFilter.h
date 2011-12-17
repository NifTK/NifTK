/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-14 11:37:54 +0100 (Wed, 14 Sep 2011) $
 Revision          : $Revision: 7310 $
 Last modified by  : $Author: ad $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef __itkCheckForThreeLevelsFilter_h
#define __itkCheckForThreeLevelsFilter_h

#include "itkImage.h"
#include "itkBaseCTESegmentationFilter.h"


namespace itk
{
/**
 * \class CheckForThreeLevelsFilter
 * \brief Simply checks for 3 levels and passes data straight through.
 * 
 * So, this is the default case of what you can do to the segmented volume
 * before passing it onto a cortical thickness pipeline. i.e. nothing.
 * All we actually do is check that the segmented volume contains 3 values.
 * Once the filter has run, we can expose a list of indexes to 
 * grey matter voxels, just in case anyone wants them. In practice,
 * its probably just as simple for each class in a pipeline to try
 * and be self sufficient, and work out the list themselves
 * from their own inputs.
 * 
 */
template <class TImageType>
class ITK_EXPORT CheckForThreeLevelsFilter : 
    public BaseCTESegmentationFilter< TImageType >
{
public:
  /** Standard "Self" & Superclass typedef.   */
  typedef CheckForThreeLevelsFilter                  Self;
  typedef BaseCTESegmentationFilter<TImageType>      Superclass;
  typedef SmartPointer<Self>                         Pointer;
  typedef SmartPointer<const Self>                   ConstPointer;

  /** Method for creation through the object factory.  */
  itkNewMacro(Self);

  /** Run-time type information (and related methods)  */
  itkTypeMacro(CheckForThreeLevelsFilter, BaseCTESegmentationFilter);
  
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
  
  /** Once the filter has run, a list of GM voxels in input segmented volume */
  const std::vector<InputIndexType>& GetListOfGreyMatterPixels() const { return this->m_ListOfGreyMatterVoxels; }

  /** Sets the segmented image, at input 0. */
  void SetSegmentedImage(const InputImageType *image) {this->SetNthInput(0, const_cast<InputImageType *>(image)); }
      
protected:
  
  CheckForThreeLevelsFilter();
  virtual ~CheckForThreeLevelsFilter()  {};

  /** Standard Print Self. */
  virtual void PrintSelf(std::ostream&, Indent) const;

  /* The main filter method. Note, single threaded. */
  virtual void GenerateData();
  
private:

  CheckForThreeLevelsFilter(const Self&); // purposely not implemented
  void operator=(const Self&); // purposely not implemented

  std::vector<InputIndexType> m_ListOfGreyMatterVoxels;
};
  
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkCheckForThreeLevelsFilter.txx"
#endif

#endif
