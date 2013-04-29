/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkBaseCTEFilter_h
#define __itkBaseCTEFilter_h

#include "itkImage.h"
#include "itkImageToImageFilter.h"

namespace itk
{
/**
 * \class BaseCTEFilter
 * \brief Base class for methods many CTE filters will need.
 * 
 * For simplicity (in this ITK template ridden world), we assume that
 * the input image and output image are the same type. This means, we
 * only need one templated type, but it means that the client needs
 * to instantiate the correct type, and be consistent in their application
 * program. For example, no point using PV maps of shorts, to represent floats,
 * as all the probabilities (between 0 and 1) will get rounded to exactly 0 or 1.
 * 
 */
template <class TImageType>
class ITK_EXPORT BaseCTEFilter : 
    public ImageToImageFilter< TImageType, TImageType >
{
public:
  /** Standard "Self" & Superclass typedef.   */
  typedef BaseCTEFilter                                       Self;
  typedef ImageToImageFilter< TImageType, TImageType >        Superclass;
  typedef SmartPointer<Self>                                  Pointer;
  typedef SmartPointer<const Self>                            ConstPointer;

  /** Run-time type information (and related methods)  */
  itkTypeMacro(BaseCTEFilter, ImageToImageFilter);
  
  /** So we can use the image Dimension as a variable. */
  itkStaticConstMacro(Dimension, unsigned int, TImageType::ImageDimension);
  
  /** Image typedef support. */
  typedef TImageType                                          InputImageType;
  typedef typename InputImageType::PixelType                  InputPixelType;
  typedef typename InputImageType::IndexType                  InputIndexType;
  typedef typename InputImageType::SizeType                   InputSizeType;
  typedef typename InputImageType::Pointer                    InputImagePointer;
  typedef typename InputImageType::ConstPointer               InputImageConstPointer;
  typedef InputPixelType                                      OutputPixelType;
  typedef Image<OutputPixelType, TImageType::ImageDimension>  OutputImageType;
  typedef typename OutputImageType::Pointer                   OutputImagePointer;
  typedef typename OutputImageType::ConstPointer              OutputImageConstPointer;
  typedef typename OutputImageType::SizeType                  OutputSizeType;
  
  /** Get the Grey Matter label. */
  itkGetMacro( GreyMatterLabel, InputPixelType);

  /** Get the White Matter label. */
  itkGetMacro( WhiteMatterLabel, InputPixelType);

  /** Get the CSF label. */
  itkGetMacro( ExtraCerebralMatterLabel, InputPixelType);

  /** This forces you to set all 3 at once. */
  void SetLabelThresholds(InputPixelType greyMatterLabel,
                          InputPixelType whiteMatterLabel,
                          InputPixelType extraCerebralMatterLabel);

  /** So we can check if the user set the thresholds, or they were worked out automatically. */
  itkGetMacro(UserHasSetTheLabelThresholds, bool);

protected:
  
  BaseCTEFilter();
  virtual ~BaseCTEFilter()  {};

  /** Standard Print Self. */
  virtual void PrintSelf(std::ostream&, Indent) const;

  /** Force the filter to request LargestPossibleRegion on input. */
  virtual void GenerateInputRequestedRegion();

  /** Force filter to create the output buffer at LargestPossibleRegion */
  virtual void EnlargeOutputRequestedRegion(DataObject *itkNotUsed);

  /** Checks all inputs and outputs are the same size. */
  virtual void CheckInputsAndOutputsSameSize();
  
  /** Check for a boundary. */
  virtual bool IsOnBoundary(const InputImageType *image, const InputIndexType& index, const InputPixelType boundaryValue, bool useFullyConnected);
  
  /** Check if we are on CSF boundary. */
  virtual bool IsOnCSFBoundary(const InputImageType *image, const InputIndexType& index, bool useFullyConnected) 
    { return this->IsOnBoundary(image, index, this->m_ExtraCerebralMatterLabel, useFullyConnected);}    

  /** Check if we are on WM boundary. */
  virtual bool IsOnWMBoundary(const InputImageType *image, const InputIndexType& index, bool useFullyConnected) 
    { return this->IsOnBoundary(image, index, this->m_WhiteMatterLabel, useFullyConnected);}    

  /** Check if we are on GM boundary. */
  virtual bool IsOnGMBoundary(const InputImageType *image, const InputIndexType& index, bool useFullyConnected) 
    { return this->IsOnBoundary(image, index, this->m_GreyMatterLabel, useFullyConnected);}    

  InputPixelType m_GreyMatterLabel;
  
  InputPixelType m_WhiteMatterLabel;
  
  InputPixelType m_ExtraCerebralMatterLabel;
  
  bool m_UserHasSetTheLabelThresholds;

private:

  BaseCTEFilter(const Self&); // purposely not implemented
  void operator=(const Self&); // purposely not implemented
    
};
  
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkBaseCTEFilter.txx"
#endif

#endif
