/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
#ifndef ITKINTENSITYFILTER_H
#define ITKINTENSITYFILTER_H

#include <itkImageToImageFilter.h>
#include <itkMacro.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkNormalizeImageFilter.h>

namespace itk {

/** \class IntensityFilter
 * \brief Uses intensity information to enhance the vesselness filter
 * response.
 */
template < class TIntensityImage, class TVesselImage >
class ITK_EXPORT IntensityFilter :
    public ImageToImageFilter< TIntensityImage, TVesselImage >
{
public:
  /** Standard class typedefs. */
  typedef IntensityFilter                                    Self;
  typedef ImageToImageFilter<TIntensityImage, TVesselImage>  Superclass;
  typedef SmartPointer<Self>                                 Pointer;
  typedef SmartPointer<const Self>                           ConstPointer;
  typedef TIntensityImage                                    IntensityImageType;
  typedef TVesselImage                                       VesselImageType;
  typedef typename IntensityImageType::PixelType             OutputPixelType;
  typedef OutputPixelType                                    InternalPixelType;

  typedef enum
  {
    LINEAR = 0,
    EXPONENTIAL = 1,
    MULTIPLY = 2
  } FilterModeType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(IntensityFilter, ImageToImageFilter);

  itkStaticConstMacro(ImageDimension, unsigned int, TIntensityImage::ImageDimension);

  void SetIntensityImage(const TIntensityImage* image);
  void SetVesselnessImage(const TVesselImage* image);

  itkGetConstMacro(FilterMode,    FilterModeType);
  itkGetConstMacro(Degree,        InternalPixelType);
  itkGetConstMacro(Threshold,     InternalPixelType);
  itkGetConstMacro(OutputMaximum, InternalPixelType);

  itkSetMacro(FilterMode,     FilterModeType);
  itkSetMacro(Degree,         InternalPixelType);
  itkSetMacro(Threshold,      InternalPixelType);
  itkSetMacro(OutputMaximum,  InternalPixelType);

protected:
  IntensityFilter();
  ~IntensityFilter(){}

  typedef Image<InternalPixelType, ImageDimension>           InternalImageType;
  typedef itk::RescaleIntensityImageFilter< VesselImageType,
                                        InternalImageType > VesselRescalerType;
  typedef itk::RescaleIntensityImageFilter< IntensityImageType,
                                          InternalImageType >InputRescalerType;
  typedef itk::RescaleIntensityImageFilter< InternalImageType,
                                          InternalImageType >InternalRescalerType;
  typedef itk::CastImageFilter< InternalImageType, VesselImageType >
                                                          CastOutFilterType;
  typedef itk::NormalizeImageFilter< IntensityImageType,
                                          InternalImageType >NormalizerIntensityType;
  typedef itk::NormalizeImageFilter< VesselImageType,
                                            InternalImageType >NormalizerVesselType;


  typename IntensityImageType::ConstPointer GetIntensityImage();
  typename VesselImageType::ConstPointer    GetVesselnessImage();

  /** Does the real work. */
  virtual void GenerateData();

private:
  IntensityFilter(const Self &); //purposely not implemented
  void operator=(const Self &);  //purposely not implemented
  void PrintSelf(std::ostream&os, Indent indent) const;

  FilterModeType    m_FilterMode;
  InternalPixelType m_Degree;
  InternalPixelType m_Threshold;
  InternalPixelType m_OutputMaximum;
};
} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkIntensityFilter.txx"
#endif

#endif // ITKINTENSITYFILTER_H
