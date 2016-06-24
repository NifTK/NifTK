/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
#ifndef ITKRESAMPLEIMAGE_H
#define ITKRESAMPLEIMAGE_H

#include <itkIdentityTransform.h>
#include <itkBSplineInterpolateImageFunction.h>
#include <itkResampleImageFilter.h>
#include <itkNearestNeighborInterpolateImageFunction.h>
#include <itkImageToImageFilter.h>
#include <itkMacro.h>

namespace itk {

/** \class ResampleImage
 * \brief Up/Down samples an image in the axial direction to reduce anistoropy
 */
template < class TInputImage >
class ResampleImage :
public ImageToImageFilter<TInputImage, TInputImage>
{
public:
  /** Standard class typedefs. */
  typedef ResampleImage                  Self;
  typedef ImageToImageFilter<TInputImage,TInputImage>  Superclass;
  typedef SmartPointer<Self>                            Pointer;
  typedef SmartPointer<const Self>                      ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ResampleImage, ImageToImageFilter);

  itkStaticConstMacro(ImageDimension, unsigned int, TInputImage::ImageDimension);

  /** Inherit types from Superclass. */
  typedef typename Superclass::InputImageType         InputImageType;
  typedef typename Superclass::OutputImageType        OutputImageType;
  typedef typename Superclass::InputImagePointer      InputImagePointer;
  typedef typename Superclass::OutputImagePointer     OutputImagePointer;
  typedef typename Superclass::InputImageConstPointer InputImageConstPointer;

  itkGetConstMacro(AxialSpacing, double);
  itkGetConstMacro(AxialSize, unsigned int);
  itkSetMacro(AxialSpacing, double);
  itkSetMacro(AxialSize, unsigned int);

protected:
  ResampleImage();
  ~ResampleImage() {};
  void PrintSelf(std::ostream&os, Indent indent) const;
  /** Does the real work. */
  virtual void GenerateData();

  typedef itk::IdentityTransform<double, 3> TransformType;
  //typedef itk::NearestNeighborInterpolateImageFunction< InputImageType, double > // Possibly used in the coming back
  typedef itk::BSplineInterpolateImageFunction<InputImageType, double, double>
      InterpolatorType;
  typedef itk::ResampleImageFilter< InputImageType, OutputImageType >
      ResampleFilterType;

private:
  ResampleImage(const Self &); //purposely not implemented
  void operator=(const Self &);  //purposely not implemented

  //bool    m_Downsample;
  double  m_AxialSpacing;
  unsigned int m_AxialSize;
};

}
#ifndef ITK_MANUAL_INSTANTIATION
#include "itkResampleImage.txx"
#endif

#endif // ITKRESAMPLEIMAGE_H
