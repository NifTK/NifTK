/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkBinaryShapeBasedSuperSamplingFilter_h
#define itkBinaryShapeBasedSuperSamplingFilter_h

#include <itkSampleImageFilter.h>


namespace itk {
  
/** \class BinaryShapeBasedSuperSamplingFilter
 * \brief Filter to super-sample a mask by a certain factor
 * and apply the appropriate shape based interpolation.
 *
 */
template < class TInputImage, class TOutputImage >
class ITK_EXPORT BinaryShapeBasedSuperSamplingFilter : 
    public SampleImageFilter< TInputImage, TOutputImage >
{
public:
  /** Standard class typedefs. */
  typedef BinaryShapeBasedSuperSamplingFilter           Self;
  typedef SampleImageFilter<TInputImage,TOutputImage>   Superclass;
  typedef SmartPointer<Self>                            Pointer;
  typedef SmartPointer<const Self>                      ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(BinaryShapeBasedSuperSamplingFilter, SampleImageFilter);

  /** ImageDimension enumeration. */
  itkStaticConstMacro(ImageDimension, unsigned int, TInputImage::ImageDimension);
  itkStaticConstMacro(OutputImageDimension, unsigned int, TOutputImage::ImageDimension);

  /** Inherit types from Superclass. */
  typedef typename Superclass::InputImageType         InputImageType;
  typedef typename Superclass::OutputImageType        OutputImageType;
  typedef typename Superclass::InputImagePointer      InputImagePointer;
  typedef typename Superclass::OutputImagePointer     OutputImagePointer;
  typedef typename Superclass::InputImageConstPointer InputImageConstPointer;

  typedef typename itk::Image<unsigned char, ImageDimension> MaskImageType;
  typedef typename MaskImageType::Pointer MaskImagePointer;

  typedef typename itk::Image<float, ImageDimension> FloatImageType; 
  typedef typename FloatImageType::Pointer FloatImagePointer;


protected:
  BinaryShapeBasedSuperSamplingFilter();
  ~BinaryShapeBasedSuperSamplingFilter() {};
  void PrintSelf(std::ostream&os, Indent indent) const;

  /** Generate the output data. */
  virtual void GenerateData();

  FloatImagePointer SmoothDistanceMap( unsigned int idim, FloatImagePointer image );

private:
  BinaryShapeBasedSuperSamplingFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
  
};


} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkBinaryShapeBasedSuperSamplingFilter.txx"
#endif

#endif
