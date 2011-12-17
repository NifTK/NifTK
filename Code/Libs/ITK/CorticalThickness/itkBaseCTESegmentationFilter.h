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
#ifndef __itkBaseCTESegmentationFilter_h
#define __itkBaseCTESegmentationFilter_h

#include "itkImage.h"
#include "itkBaseCTEFilter.h"


namespace itk
{
/**
 * \class BaseCTESegmentationFilter
 * \brief Base class for classes that manipulate the segmented volume
 * before it gets to the Cortical Thickness Estimation.
 * 
 * For example, as of right now, we have two derived classes.
 * CheckForThreeLevelsFilter simply checks that the input segmented
 * volume has exactly three labels, but passes data straight through,
 * whereas the CorrectGMUsingPVMapFilter does the same check for
 * three levels, and then modifies the grey matter according to the
 * grey matter PV map. Hence the code for checking for three levels
 * is in this class.
 * 
 * \sa CorrectGMUsingPVMapFilter
 * \sa CheckForThreeLevelsFilter 
 */
template <class TImageType>
class ITK_EXPORT BaseCTESegmentationFilter : 
    public BaseCTEFilter< TImageType >
{
public:
  /** Standard "Self" & Superclass typedef.   */
  typedef BaseCTESegmentationFilter       Self;
  typedef BaseCTEFilter< TImageType >     Superclass;
  typedef SmartPointer<Self>              Pointer;
  typedef SmartPointer<const Self>        ConstPointer;

  /** Run-time type information (and related methods)  */
  itkTypeMacro(BaseCTESegmentationFilter, BaseCTEFilter);
  
  /** Image typedef support. */
  typedef typename Superclass::InputImageType          InputImageType;
  typedef typename Superclass::InputPixelType          InputPixelType;
  typedef typename Superclass::InputIndexType          InputIndexType;
  typedef typename Superclass::InputSizeType           InputSizeType;
  typedef typename Superclass::InputImagePointer       InputImagePointer;
  typedef typename Superclass::InputImageConstPointer  InputImageConstPointer;
  typedef typename Superclass::OutputPixelType         OutputPixelType;
  typedef typename Superclass::OutputImageType         OutputImageType;
  typedef typename Superclass::OutputImagePointer      OutputImagePointer;
  typedef typename Superclass::OutputImageConstPointer OutputImageConstPointer;
  typedef typename Superclass::OutputSizeType          OutputSizeType;
  
  /** If this is false, we check each PV map to see if the values are >= 0 and <= 1. Defaults to false. */
  itkSetMacro(TrustPVMaps, bool);
  itkGetMacro(TrustPVMaps, bool);

protected:
  
  BaseCTESegmentationFilter();
  virtual ~BaseCTESegmentationFilter()  {};

  /** Standard Print Self. */
  virtual void PrintSelf(std::ostream&, Indent) const;

  /** 
   * Checks that we have specified 3 labels (GM, WM, CSF), and that the input
   * image contains JUST those 3 values, or if they are unspecified, we 
   * calculate them by counting how many of each pixel there are, and assigning
   * the most to CSF, the least to GM, and the middle quantity to WM.
   */
  virtual void CheckOrAssignLabels();
  
  /** 
   * If m_TrustPVMaps is false, we whizz through image, checking that all 
   * values are >= 0 and <= 1. If they are not an exception is thrown.
   * If m_TrustPVMaps is true, this method does nothing. 
   */
  virtual void CheckPVMap(std::string name, const InputImageType *image);

private:

  BaseCTESegmentationFilter(const Self&); // purposely not implemented
  void operator=(const Self&); // purposely not implemented
    
  bool m_TrustPVMaps;

};
  
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkBaseCTESegmentationFilter.txx"
#endif

#endif
