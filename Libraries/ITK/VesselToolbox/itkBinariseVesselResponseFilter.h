/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef ITKBINARISEVESSELRESPONSEFILTER_H
#define ITKBINARISEVESSELRESPONSEFILTER_H

#include <itkImageToImageFilter.h>
#include <itkMacro.h>
#include <itkRelabelComponentImageFilter.h>
#include <itkConnectedThresholdImageFilter.h>
#include <itkNormalizeImageFilter.h>
#include <itkBinaryThresholdImageFilter.h>

namespace itk {
/** \class BinariseVesselResponseFilter
 * \brief Binarises the vesselness response and keeps the largest objects.
 */
template < class TInputImage, class TOutputImage >
class ITK_EXPORT BinariseVesselResponseFilter :
    public ImageToImageFilter< TInputImage, TOutputImage >
{
public:
  /** Standard class typedefs. */
  typedef BinariseVesselResponseFilter                  Self;
  typedef ImageToImageFilter<TInputImage, TOutputImage> Superclass;
  typedef SmartPointer<Self>                            Pointer;
  typedef SmartPointer<const Self>                      ConstPointer;

  typedef short                                         InternalPixelType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(BinariseVesselResponseFilter, ImageToImageFilter);

  itkStaticConstMacro(ImageDimension, unsigned int, TInputImage::ImageDimension);

  /** Inherit types from Superclass. */
  typedef typename Superclass::InputImageType         InputImageType;
  typedef typename Superclass::OutputImageType        OutputImageType;
  typedef typename Superclass::InputImagePointer      InputImagePointer;
  typedef typename Superclass::OutputImagePointer     OutputImagePointer;
  typedef typename Superclass::InputImageConstPointer InputImageConstPointer;
  typedef typename InputImageType::PixelType          InputPixelType;
  typedef typename OutputImageType::PixelType         OutputPixelType;

  itkGetConstMacro(LowThreshold, InternalPixelType);
  itkGetConstMacro(UpThreshold, InternalPixelType);
  itkGetConstMacro(Percentage, float);
  itkSetMacro(LowThreshold, InternalPixelType);
  itkSetMacro(UpThreshold, InternalPixelType);
  itkSetMacro(Percentage, float);

protected:
  BinariseVesselResponseFilter();
  ~BinariseVesselResponseFilter() {};
  void PrintSelf(std::ostream&os, Indent indent) const;

  /** Generate the output data. */
  virtual void GenerateData();

  typedef Image<InternalPixelType, ImageDimension>                              InternalImageType;
  typedef itk::NormalizeImageFilter<InputImageType,
                                     InternalImageType>                         NormalizerType;
  typedef itk::BinaryThresholdImageFilter<InternalImageType, InternalImageType> ThresholdFilter;
  typedef itk::ConnectedComponentImageFilter <InternalImageType, InternalImageType>
                                                            ConnectedComponentImageFilterType;
  typedef itk::RelabelComponentImageFilter<InternalImageType, OutputImageType>  RelabelFilterType;

private:
  BinariseVesselResponseFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  InternalPixelType  m_LowThreshold;
  InternalPixelType  m_UpThreshold;
  float              m_Percentage;
};

} //end namespace

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkBinariseVesselResponseFilter.txx"
#endif

#endif // ITKBINARISEVESSELRESPONSEFILTER_H
