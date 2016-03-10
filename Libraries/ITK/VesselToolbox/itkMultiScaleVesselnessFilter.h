/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
#ifndef ITKMULTISCALEVESSELNESSFILTER_H
#define ITKMULTISCALEVESSELNESSFILTER_H

#include <itkImageToImageFilter.h>
#include <itkMacro.h>
#include <itkCastImageFilter.h>
#include <itkHessianRecursiveGaussianImageFilter.h>
#include <itkHessian3DToVesselnessMeasureImageFilter.h>
#include <math.h>

namespace itk {

/** \class MultiScaleVesselnessFilter
 * \brief Gives tha maximum filter response using Sato's filter
 * (Sato et al, MedIA 1998) per voxel, given a range of scales
 */
template < class TInputImage, class TOutputImage >
class ITK_EXPORT MultiScaleVesselnessFilter :
    public ImageToImageFilter< TInputImage, TOutputImage >
{
public:
  /** Standard class typedefs. */
  typedef MultiScaleVesselnessFilter                          Self;
  typedef ImageToImageFilter<TInputImage,TOutputImage>  Superclass;
  typedef SmartPointer<Self>                            Pointer;
  typedef SmartPointer<const Self>                      ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(MultiScaleVesselnessFilter, ImageToImageFilter);

  itkStaticConstMacro(ImageDimension, unsigned int, TInputImage::ImageDimension);

  /** Inherit types from Superclass. */
  typedef typename Superclass::InputImageType         InputImageType;
  typedef typename Superclass::OutputImageType        OutputImageType;
  typedef typename Superclass::InputImagePointer      InputImagePointer;
  typedef typename Superclass::OutputImagePointer     OutputImagePointer;
  typedef typename Superclass::InputImageConstPointer InputImageConstPointer;
  typedef typename InputImageType::SpacingType        SpacingType;
  typedef typename OutputImageType::PixelType         OutputPixelType;

  typedef enum
  {
    LINEAR = 0,
    EXPONENTIAL = 1
  } ScaleModeType;


  itkGetConstMacro(AlphaOne, float);
  itkGetConstMacro(AlphaTwo, float);
  itkGetConstMacro(MinScale, float);
  itkGetConstMacro(MaxScale, float);
  itkGetConstMacro(ScaleMode, ScaleModeType);
  itkSetMacro(AlphaOne, float);
  itkSetMacro(AlphaTwo, float);
  itkSetMacro(MinScale, float);
  itkSetMacro(MaxScale, float);
  itkSetMacro(ScaleMode, ScaleModeType);

protected:
  MultiScaleVesselnessFilter();
  ~MultiScaleVesselnessFilter() { };
  void PrintSelf(std::ostream&os, Indent indent) const;

  typedef itk::CastImageFilter< InputImageType, OutputImageType > CastFilterType;
  //typedef itk::CastImageFilter< VesselImageType, OutputImageType > CastOutFilterType;
  typedef itk::HessianRecursiveGaussianImageFilter< OutputImageType > HessianFilterType;
  typedef itk::Hessian3DToVesselnessMeasureImageFilter< OutputPixelType > VesselnessMeasureFilterType;


  /** Generate the output data. */
  virtual void GenerateData();

private:
  MultiScaleVesselnessFilter(const Self &); //purposely not implemented
  void operator=(const Self &);  //purposely not implemented


  float m_AlphaOne;
  float m_AlphaTwo;
  float   m_MinScale;
  float   m_MaxScale;
  ScaleModeType m_ScaleMode;
};

}

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMultiScaleVesselnessFilter.txx"
#endif


#endif // ITKMULTISCALEVESSELNESSFILTER_H
