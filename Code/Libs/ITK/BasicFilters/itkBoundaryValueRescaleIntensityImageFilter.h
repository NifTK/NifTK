/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkBoundaryValueRescaleIntensityImageFilter_h
#define __itkBoundaryValueRescaleIntensityImageFilter_h
#include "itkImageToImageFilter.h"
#include "itkBinaryThresholdImageFilter.h"

namespace itk
{
/** \class BoundaryValueRescaleIntensityImageFilter
 * \brief Applies a linear transformation to the intensity levels of the
 * input Image, but takes into an upper and lower threshold, so we can reassign
 * stuff thats above or below (respectively) these thresholds to a set
 * OutputBoundaryValue.
 * 
 * The user should specify InputLowerThreshold, InputUpperThreshold,
 * OutputBoundaryValue, OutputMinimum, OutputMaximum.
 * 
 * The output is then
 * <pre>
 * if 
 *   input < InputLowerThreshold, output = OutputBoundaryValue
 * else if 
 *   input > InputUpperThreshold, output = OutputBoundaryValue
 * else 
 *   the max and minimum of the input image is calculated, and each value
 *   rescaled to match the range between OutputMinimum and OutputMaximum
 * </pre>
 * \ingroup IntensityImageFilters Multithreaded
 *
 */
template <typename  TImageType>
class ITK_EXPORT BoundaryValueRescaleIntensityImageFilter : public ImageToImageFilter<TImageType, TImageType>
{
public:
  /** Standard class typedefs. */
  typedef BoundaryValueRescaleIntensityImageFilter            Self;
  typedef ImageToImageFilter<TImageType, TImageType>          Superclass;
  typedef SmartPointer<Self>                                  Pointer;
  typedef SmartPointer<const Self>                            ConstPointer;
  typedef typename TImageType::PixelType                      PixelType;
  typedef TImageType                                          ImageType;
  typedef typename ImageType::Pointer                         ImagePointer;
  typedef typename ImageType::RegionType                      ImageRegionType;
  typedef BinaryThresholdImageFilter<ImageType, ImageType>    BinaryThresholdFilterType;
  typedef typename BinaryThresholdFilterType::Pointer         BinaryThresholdFilterPointer;
  typedef typename NumericTraits<PixelType>::RealType         RealType;
  
  /** Method for creation through the object factory. */
  itkNewMacro(Self);
  
  /** Runtime information support. */
  itkTypeMacro(BoundaryValueRescaleIntensityImageFilter, ImageToImageFilter);

  /** Print internal ivars */
  void PrintSelf(std::ostream& os, Indent indent) const;

  /** Set/Get the lower threshold, values < this are masked out, and set to OutputBoundaryValue. */
  itkSetMacro(InputLowerThreshold, PixelType);
  itkGetMacro(InputLowerThreshold, PixelType);

  /** Set/Get the upper threshold, values > this are masked out, and set to OutputBoundaryValue. */
  itkSetMacro(InputUpperThreshold, PixelType);
  itkGetMacro(InputUpperThreshold, PixelType);

  /** Set/Get the output minimum (apart from OutputBoundaryValue). */
  itkSetMacro(OutputMinimum, PixelType);
  itkGetMacro(OutputMinimum, PixelType);

  /** Set/Get the output maximum (apart from OutputBoundaryValue). */
  itkSetMacro(OutputMaximum, PixelType);
  itkGetMacro(OutputMaximum, PixelType);

  /** Set/Get the output boundary value. Not included in rescale calculations. */
  itkSetMacro(OutputBoundaryValue, PixelType);
  itkGetMacro(OutputBoundaryValue, PixelType);

protected:
  
  BoundaryValueRescaleIntensityImageFilter();
  virtual ~BoundaryValueRescaleIntensityImageFilter() {};

  // Check before we start.
  virtual void BeforeThreadedGenerateData();
  
  // The main method to implement in derived classes, note, its threaded.
  virtual void ThreadedGenerateData( const ImageRegionType &outputRegionForThread, int);

private:
  BoundaryValueRescaleIntensityImageFilter(const Self&); // purposely not implemented
  void operator=(const Self&); // purposely not implemented

  RealType        m_Scale;
  RealType        m_Shift;

  PixelType       m_InputLowerThreshold;
  PixelType       m_InputUpperThreshold;
  
  PixelType       m_InputMinimum;
  PixelType       m_InputMaximum;

  PixelType       m_OutputMinimum;
  PixelType       m_OutputMaximum;
  PixelType       m_OutputBoundaryValue;

  /** use standard ITK filter for thresholding. */
  BinaryThresholdFilterPointer m_ThresholdFilter;
  
};


  
} // end namespace itk
  
#ifndef ITK_MANUAL_INSTANTIATION
#include "itkBoundaryValueRescaleIntensityImageFilter.txx"
#endif
  
#endif
