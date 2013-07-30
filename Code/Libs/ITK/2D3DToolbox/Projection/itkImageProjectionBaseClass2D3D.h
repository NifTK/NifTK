/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkImageProjectionBaseClass2D3D_h
#define itkImageProjectionBaseClass2D3D_h

#include <itkImageToImageFilter.h>
#include <itkConceptChecking.h>
#include <itkPerspectiveProjectionTransform.h>
#include <itkEulerAffineTransform.h>

namespace itk
{
  
/** \class ImageProjectionBaseClass2D3D
 * \brief The base class for 2D-3D forward and back projection.
 */

template <class TInputImage, class TOutputImage>
class ITK_EXPORT ImageProjectionBaseClass2D3D : 
  public ImageToImageFilter<TInputImage,TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef ImageProjectionBaseClass2D3D                  Self;
  typedef ImageToImageFilter<TInputImage,TOutputImage>  Superclass;
  typedef SmartPointer<Self>                            Pointer;
  typedef SmartPointer<const Self>                      ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ImageProjectionBaseClass2D3D, ImageToImageFilter);

  /** Some convenient typedefs. */
  typedef TInputImage                              InputImageType;
  typedef typename    InputImageType::Pointer      InputImagePointer;
  typedef typename    InputImageType::RegionType   InputImageRegionType;
  typedef typename    InputImageType::PixelType    InputImagePixelType;

  typedef TOutputImage                             OutputImageType;
  typedef typename     OutputImageType::Pointer    OutputImagePointer;
  typedef typename     OutputImageType::RegionType OutputImageRegionType;
  typedef typename     OutputImageType::PixelType  OutputImagePixelType;

  typedef EulerAffineTransform<double, 3, 3> EulerAffineTransformType;
  typedef PerspectiveProjectionTransform<double> PerspectiveProjectionTransformType;


  /** ImageDimension enumeration */
  itkStaticConstMacro(InputImageDimension, unsigned int,
                      TInputImage::ImageDimension);
  itkStaticConstMacro(OutputImageDimension, unsigned int,
                      TOutputImage::ImageDimension);

  /// Set the affine transformation
  itkSetObjectMacro( AffineTransform, EulerAffineTransformType );
  /// Get the affine transformation
  itkGetObjectMacro( AffineTransform, EulerAffineTransformType );
  /// Set the perspective transformation
  itkSetObjectMacro( PerspectiveTransform, PerspectiveProjectionTransformType );
  /// Get the perspective transformation
  itkGetObjectMacro( PerspectiveTransform, PerspectiveProjectionTransformType );

protected:
  ImageProjectionBaseClass2D3D();
  virtual ~ImageProjectionBaseClass2D3D() {};
  void PrintSelf(std::ostream& os, Indent indent) const;

  /// The affine transform
  EulerAffineTransformType::Pointer m_AffineTransform; 

  // The perspective transform
  PerspectiveProjectionTransformType::Pointer m_PerspectiveTransform;

private:
  ImageProjectionBaseClass2D3D(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkImageProjectionBaseClass2D3D.txx"
#endif

#endif
