/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkSubsampleImageFilter_h
#define __itkSubsampleImageFilter_h

#include <itkImageToImageFilter.h>
#include <itkMacro.h>


namespace itk {
  
/** \class SubsampleImageFilter
 * \brief Filter to subsample an image by a certain factor and apply
 * the appropriate blurring (equivalent to voxel averaging for integer
 * subsampling factors).
 *
 */
template < class TInputImage, class TOutputImage >
class ITK_EXPORT SubsampleImageFilter : 
    public ImageToImageFilter< TInputImage, TOutputImage >
{
public:
  /** Standard class typedefs. */
  typedef SubsampleImageFilter                          Self;
  typedef ImageToImageFilter<TInputImage,TOutputImage>  Superclass;
  typedef SmartPointer<Self>                            Pointer;
  typedef SmartPointer<const Self>                      ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(SubsampleImageFilter, ImageToImageFilter);
/*
  itkLogMacro(DEBUG,     "DEBUG message by itkLogMacro\n");
  itkLogMacro(INFO,      "INFO message by itkLogMacro\n");
  itkLogMacro(WARNING,   "WARNING message by itkLogMacro\n");
  itkLogMacro(CRITICAL,  "CRITICAL message by itkLogMacro\n");
  itkLogMacro(FATAL,     "FATAL message by itkLogMacro\n");
  itkLogMacro(MUSTFLUSH, "MUSTFLUSH message by itkLogMacro\n");
*/
  /** ImageDimension enumeration. */
  itkStaticConstMacro(ImageDimension, unsigned int, TInputImage::ImageDimension);
  itkStaticConstMacro(OutputImageDimension, unsigned int, TOutputImage::ImageDimension);

  /** Inherit types from Superclass. */
  typedef typename Superclass::InputImageType         InputImageType;
  typedef typename Superclass::OutputImageType        OutputImageType;
  typedef typename Superclass::InputImagePointer      InputImagePointer;
  typedef typename Superclass::OutputImagePointer     OutputImagePointer;
  typedef typename Superclass::InputImageConstPointer InputImageConstPointer;

  /** Directly Set/Get the array of subsampling factors for each image dimension */
  void SetSubsamplingFactors(double data[]);
  itkGetVectorMacro(SubsamplingFactors, const double, ::itk::GetImageDimension<TInputImage>::ImageDimension);

  /** SubsampleImageFilter produces images which are of
   * different resolution and different pixel spacing than its input image.
   * As such, SubsampleImageFilter needs to provide an
   * implementation for GenerateOutputInformation() in order to inform the
   * pipeline execution model.  The original documentation of this method is
   * below.  \sa ProcessObject::GenerateOutputInformaton() */
  virtual void GenerateOutputInformation();

  /** SubsampleImageFilter requires a larger input requested
   * region than the output requested regions to accomdate the shrinkage and
   * smoothing operations. As such, SubsampleImageFilter needs
   * to provide an implementation for GenerateInputRequestedRegion().  The
   * original documentation of this method is below.  \sa
   * ProcessObject::GenerateInputRequestedRegion() */
  virtual void GenerateInputRequestedRegion();

  itkSetMacro(MaximumError,double);
  itkGetConstReferenceMacro(MaximumError,double);


#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro(SameDimensionCheck,
    (Concept::SameDimension<ImageDimension, OutputImageDimension>));
  itkConceptMacro(OutputHasNumericTraitsCheck,
    (Concept::HasNumericTraits<typename TOutputImage::PixelType>));
  /** End concept checking */
#endif

protected:
  SubsampleImageFilter();
  ~SubsampleImageFilter() {};
  void PrintSelf(std::ostream&os, Indent indent) const;

  /** Generate the output data. */
  void GenerateData();

  double          m_MaximumError; 

  double m_SubsamplingFactors[::itk::GetImageDimension<TInputImage>::ImageDimension];

private:
  SubsampleImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
  
};


} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkSubsampleImageFilter.txx"
#endif

#endif
