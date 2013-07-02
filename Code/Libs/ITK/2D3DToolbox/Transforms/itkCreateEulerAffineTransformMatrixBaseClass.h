/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkCreateEulerAffineTransformMatrixBaseClass_h
#define __itkCreateEulerAffineTransformMatrixBaseClass_h

#include <itkImageToImageFilter.h>
#include <itkConceptChecking.h>
#include <itkEulerAffineTransform.h>

namespace itk
{
  
/** \class CreateEulerAffineTransformMatrixBaseClass
 * \brief The base class for 3D-3D affine transformation matrix.
 */

template <class TInputImage, class TOutputImage>
class ITK_EXPORT CreateEulerAffineTransformMatrixBaseClass : 
  public ImageToImageFilter<TInputImage,TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef CreateEulerAffineTransformMatrixBaseClass                  Self;
  typedef ImageToImageFilter<TInputImage,TOutputImage>  						 Superclass;
  typedef SmartPointer<Self>                            						 Pointer;
  typedef SmartPointer<const Self>                      						 ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(CreateEulerAffineTransformMatrixBaseClass, ImageToImageFilter);

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


  /** ImageDimension enumeration */
  itkStaticConstMacro(InputImageDimension, unsigned int,
                      TInputImage::ImageDimension);
  itkStaticConstMacro(OutputImageDimension, unsigned int,
                      TOutputImage::ImageDimension);

  /** Set the affine transformation */
  itkSetObjectMacro( AffineTransform, EulerAffineTransformType );
  /** Get the affine transformation */
  itkGetObjectMacro( AffineTransform, EulerAffineTransformType );

protected:
  CreateEulerAffineTransformMatrixBaseClass();
  virtual ~CreateEulerAffineTransformMatrixBaseClass() {};
  void PrintSelf(std::ostream& os, Indent indent) const;

  /** The affine transform */
  EulerAffineTransformType::Pointer m_AffineTransform;

private:
  CreateEulerAffineTransformMatrixBaseClass(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkCreateEulerAffineTransformMatrixBaseClass.txx"
#endif

#endif
