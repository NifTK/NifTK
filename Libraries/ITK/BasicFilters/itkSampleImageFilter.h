/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkSampleImageFilter_h
#define itkSampleImageFilter_h

#include <itkImageToImageFilter.h>
#include <itkImageRegistrationFactory.h>
#include <itkMacro.h>
#include <itkArray.h>


namespace itk {
  
/** \class SampleImageFilter
 * \brief Filter to sub- or super-sample an image by a certain factor
 * and apply the appropriate blurring (equivalent to voxel averaging
 * for integer subsampling factors) when subsampling.
 *
 */
template < class TInputImage, class TOutputImage >
class ITK_EXPORT SampleImageFilter : 
    public ImageToImageFilter< TInputImage, TOutputImage >
{
public:
  /** Standard class typedefs. */
  typedef SampleImageFilter                             Self;
  typedef ImageToImageFilter<TInputImage,TOutputImage>  Superclass;
  typedef SmartPointer<Self>                            Pointer;
  typedef SmartPointer<const Self>                      ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(SampleImageFilter, ImageToImageFilter);
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

  /// Set the debugging output
  void SetDebug(bool b) { itk::Object::SetDebug(b); }
  /// Set debugging output on
  void DebugOn() { this->SetDebug(true); }
  /// Set debugging output off
  void DebugOff() { this->SetDebug(false); }

  itkBooleanMacro( Verbose );
  itkGetConstMacro( Verbose, bool );
  itkSetMacro( Verbose, bool );

  itkBooleanMacro( IsotropicVoxels );
  itkGetConstMacro( IsotropicVoxels, bool );
  itkSetMacro( IsotropicVoxels, bool );

  /** Directly Set/Get the array of sampling factors for each image dimension */
  void SetSamplingFactors(double data[]);
  void SetSamplingFactors(itk::Array< double > &sampling);
  itkGetVectorMacro(SamplingFactors, const double, TInputImage::ImageDimension);

  void SetInterpolationType( InterpolationTypeEnum interp ) {
    m_Interpolation = interp;
  }


  OutputImagePointer GetSmoothedImage( unsigned int idim, 
                                       OutputImagePointer image );
  

  /** SampleImageFilter produces images which are of
   * different resolution and different pixel spacing than its input image.
   * As such, SampleImageFilter needs to provide an
   * implementation for GenerateOutputInformation() in order to inform the
   * pipeline execution model.  The original documentation of this method is
   * below.  \sa ProcessObject::GenerateOutputInformaton() */
  virtual void GenerateOutputInformation();

  /** SampleImageFilter requires a larger input requested
   * region than the output requested regions to accomdate the shrinkage and
   * smoothing operations. As such, SampleImageFilter needs
   * to provide an implementation for GenerateInputRequestedRegion().  The
   * original documentation of this method is below.  \sa
   * ProcessObject::GenerateInputRequestedRegion() */
  virtual void GenerateInputRequestedRegion();


#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro(SameDimensionCheck,
    (Concept::SameDimension<ImageDimension, OutputImageDimension>));
  itkConceptMacro(OutputHasNumericTraitsCheck,
    (Concept::HasNumericTraits<typename TOutputImage::PixelType>));
  /** End concept checking */
#endif

protected:
  SampleImageFilter();
  ~SampleImageFilter() {};
  void PrintSelf(std::ostream&os, Indent indent) const;

  bool m_Verbose;
  bool m_IsotropicVoxels;

  InterpolationTypeEnum m_Interpolation;

  /** Generate the output data. */
  virtual void GenerateData();

  double m_SamplingFactors[TInputImage::ImageDimension];

private:
  SampleImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
  
};


} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkSampleImageFilter.txx"
#endif

#endif
