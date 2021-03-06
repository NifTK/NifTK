/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkLightLineResponseImageFilter_h
#define itkLightLineResponseImageFilter_h

#include <itkImageToImageFilter.h>
#include "itkLewisGriffinRecursiveGaussianImageFilter.h"
#include <itkMaskImageFilter.h>

#include <vnl/vnl_double_2.h>

namespace itk {
  
/** \class LightLineResponseImageFilter
 * \brief 2D image filter class to compute the light line response of an image a specific scale.
 *
 */

template<class TInputImage, class TOutputImage>
class ITK_EXPORT LightLineResponseImageFilter:
    public ImageToImageFilter< TInputImage, TOutputImage >
{
public:
  /** Standard class typedefs. */
  typedef LightLineResponseImageFilter                  Self;
  typedef ImageToImageFilter< TInputImage,TOutputImage > Superclass;
  typedef SmartPointer< Self >                           Pointer;
  typedef SmartPointer< const Self >                     ConstPointer;
  
  /** Run-time type information (and related methods).   */
  itkTypeMacro( LightLineResponseImageFilter, ImageToImageFilter );

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Image dimension. */
  itkStaticConstMacro(ImageDimension, unsigned int,
                      TInputImage::ImageDimension);

  /** Type of the input image */
  typedef TInputImage                           InputImageType;
  typedef typename InputImageType::Pointer      InputImagePointer;
  typedef typename InputImageType::ConstPointer InputImageConstPointer;
  typedef typename InputImageType::RegionType   InputImageRegionType;
  typedef typename InputImageType::PixelType    InputImagePixelType;
  typedef typename InputImageType::SpacingType  InputImageSpacingType;
  typedef typename InputImageType::PointType    InputImagePointType;

  typedef typename NumericTraits<InputImagePixelType>::RealType    RealType;

  /** Type of the output image */
  typedef TOutputImage                          OutputImageType;
  typedef typename OutputImageType::Pointer     OutputImagePointer;
  typedef typename OutputImageType::RegionType  OutputImageRegionType;
  typedef typename OutputImageType::PixelType   OutputImagePixelType;
  typedef typename OutputImageType::IndexType   OutputImageIndexType;
  typedef typename OutputImageType::PointType   OutputImagePointType;

  typedef OutputImagePointType OriginType;

  /** Optional mask image */
  typedef unsigned char                                      MaskPixelType;
  typedef typename itk::Image<MaskPixelType, ImageDimension> MaskImageType;
  typedef typename MaskImageType::Pointer                    MaskImagePointer;

  typedef typename itk::MaskImageFilter< OutputImageType, MaskImageType, OutputImageType > MaskFilterType;
  typedef typename MaskFilterType::Pointer                   MaskFilterPointer;


  /** Define the image type for internal computations 
      RealType is usually 'double' in NumericTraits. 
      Here we prefer float in order to save memory.  */

  typedef float InternalRealType;

  typedef Image< InternalRealType, TInputImage::ImageDimension > RealImageType;

  typedef typename RealImageType::Pointer RealImagePointer;

  /**  Derivative filter type, it will be the first in the pipeline  */
  typedef LewisGriffinRecursiveGaussianImageFilter < InputImageType, 
                                                     RealImageType > DerivativeFilterTypeX;
  typedef LewisGriffinRecursiveGaussianImageFilter < RealImageType,  
                                                     RealImageType > DerivativeFilterTypeY;

  /**  Pointer to a derivative filter.  */
  typedef typename DerivativeFilterTypeX::OrderEnumType  DerivativeFilterOrderEnumTypeX;
  typedef typename DerivativeFilterTypeY::OrderEnumType  DerivativeFilterOrderEnumTypeY;
				        
  /**  Pointer to a derivative filtr.  e*/
  typedef typename DerivativeFilterTypeX::Pointer  DerivativeFilterPointerX;
  typedef typename DerivativeFilterTypeY::Pointer  DerivativeFilterPointerY;

  /// Set the debugging output
  void SetDebug(bool b) { itk::Object::SetDebug(b); }
  /// Set debugging output on
  void DebugOn() { this->SetDebug(true); }
  /// Set debugging output off
  void DebugOff() { this->SetDebug(false); }

  /// Set the verbose output
  void SetVerbose(bool b) { itk::Object::SetGlobalWarningDisplay(b); }
  /// Set verbose output on
  void VerboseOn() { this->SetVerbose(true); }
  /// Set verbose output off
  void VerboseOff() { this->SetVerbose(false); }

  /** Set Sigma value. */
  void SetSigma( RealType sigma );

  /** LightLineResponseImageFilter needs all of the input to produce an
   * output. Therefore, GradientRecursiveGaussianImageFilter needs to provide
   * an implementation for GenerateInputRequestedRegion in order to inform
   * the pipeline execution model.
   * \sa ImageToImageFilter::GenerateInputRequestedRegion() */
  virtual void GenerateInputRequestedRegion()
  throw( InvalidRequestedRegionError );

  /** Define which normalization factor will be used for the Gaussian */
  void SetNormalizeAcrossScale( bool normalizeInScaleSpace );
  itkGetConstMacro( NormalizeAcrossScale, bool );

  /// Set the noise suppression parameter
  itkSetMacro( Epsilon, InternalRealType );
  /// Get the noise suppression parameter
  itkGetMacro( Epsilon, InternalRealType );

  /// Set the number of orientations to quantise into
  itkSetMacro( NumberOfOrientations, unsigned int );
  /// Get the number of orientations to quantise into
  itkGetMacro( NumberOfOrientations, unsigned int );

  /// Set the origin for oriented BIFs
  void SetOrigin( OriginType o ) {
    m_Origin = o;
    m_FlagOriginSet = true;
    this->Modified();
  }
  /// Get the origin for oriented BIFs
  itkGetMacro( Origin, OriginType );

  /// Set the coefficient to flip the orientation horizontally
  void SetFlipHorizontally( void ) { flipHorizontally = -1.; this->Modified();}
  /// Set the coefficient to flip the orientation vertically
  void SetFlipVertically( void ) { flipVertically = -1.; this->Modified();}

  /// Set the local reference orientation images
  void SetLocalOrientation( RealImagePointer OrientationInX, RealImagePointer OrientationInY ) {
    m_OrientationInX = OrientationInX;
    m_OrientationInY = OrientationInY;

    m_FlagLocalOrientationSet = true;
    m_FlagOriginSet = false;
    this->Modified();
  }

  /// Set the local reference orientation image in 'y'
  itkSetObjectMacro( OrientationInY, RealImageType );

  /// Set an optional mask image
  itkSetObjectMacro( Mask, MaskImageType );

  itkGetObjectMacro( S11, RealImageType );  ///< Get a pointer to the second derivative in 'xy'
  itkGetObjectMacro( S20, RealImageType );  ///< Get a pointer to the second derivative in 'xx'
  itkGetObjectMacro( S02, RealImageType );  ///< Get a pointer to the second derivative in 'yy'

  
  itkGetObjectMacro( Orientation, RealImageType );    ///< Get a pointer to the structure orientation

  /// Write one of the derivatives to a file ( 0 < n < 6 )
  void WriteDerivativeToFile( int n, std::string filename );

#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro(DimensionShouldBe2,
		  (Concept::SameDimension<itkGetStaticConstMacro(InputImageDimension),2>));
  itkConceptMacro(InputHasNumericTraitsCheck,
                  (Concept::HasNumericTraits<InputImagePixelType>));
  itkConceptMacro(OutputHasPixelTraitsCheck,
                  (Concept::HasPixelTraits<OutputImagePixelType>));
  /** End concept checking */
#endif

  /// For debugging purposes, set single threaded execution
  void SetSingleThreadedExecution(void) {m_FlagMultiThreadedExecution = false;}

protected:
  LightLineResponseImageFilter();
  virtual ~LightLineResponseImageFilter();
  void PrintSelf(std::ostream& os, Indent indent) const;

  /** Thread-Data Structure   */
  struct LightLineResponseThreadStruct
  {
    LightLineResponseImageFilter *Filter;
  };
  
  RealImagePointer GetDerivative( DerivativeFilterOrderEnumTypeX xOrder,
				  DerivativeFilterOrderEnumTypeY yOrder );

  /** If an imaging filter needs to perform processing after the buffer
   * has been allocated but before threads are spawned, the filter can
   * can provide an implementation for BeforeThreadedGenerateData(). The
   * execution flow in the default GenerateData() method will be:
   *      1) Allocate the output buffer
   *      2) Call BeforeThreadedGenerateData()
   *      3) Spawn threads, calling ThreadedGenerateData() in each thread.
   *      4) Call AfterThreadedGenerateData()
   * Note that this flow of control is only available if a filter provides
   * a ThreadedGenerateData() method and NOT a GenerateData() method. */
  virtual void BeforeThreadedGenerateData(void);
  
  /** If an imaging filter needs to perform processing after all
   * processing threads have completed, the filter can can provide an
   * implementation for AfterThreadedGenerateData(). The execution
   * flow in the default GenerateData() method will be:
   *      1) Allocate the output buffer
   *      2) Call BeforeThreadedGenerateData()
   *      3) Spawn threads, calling ThreadedGenerateData() in each thread.
   *      4) Call AfterThreadedGenerateData()
   * Note that this flow of control is only available if a filter provides
   * a ThreadedGenerateData() method and NOT a GenerateData() method. */
  virtual void AfterThreadedGenerateData(void);
  
  /** Single threaded execution, for debugging purposes ( call
  SetSingleThreadedExecution() ) */
  void GenerateData();
  
  /** LightLineResponseImageFilter can be implemented as a multithreaded filter.
   * Therefore, this implementation provides a ThreadedGenerateData()
   * routine which is called for each processing thread. The output
   * image data is allocated automatically by the superclass prior to
   * calling ThreadedGenerateData().  ThreadedGenerateData can only
   * write to the portion of the output image specified by the
   * parameter "outputRegionForThread"
   *
   * \sa ImageToImageFilter::ThreadedGenerateData(),
   *     ImageToImageFilter::GenerateData() */
  void ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
                            ThreadIdType threadId );

  // Override since the filter produces the entire dataset
  void EnlargeOutputRequestedRegion(DataObject *output);

  /// Flag to turn multithreading on or off
  bool m_FlagMultiThreadedExecution;

  /// Flag to calculate orientated BIFs using local orientation images
  bool m_FlagLocalOrientationSet;

  /** The number of orientations to quantise into. For example the
      default value, 8, results in 45 degree quantisation (eight slope
      orientations and four line orientations). */
  unsigned int m_NumberOfOrientations;

  /** Normalize the image across scale space */
  bool m_NormalizeAcrossScale; 
  
  /// Noise suppression parameter
  InternalRealType m_Epsilon; 

  /// Scale 
  InternalRealType m_Sigma; 

  /// Flag indicating whether the orgin has been set
  bool m_FlagOriginSet;
  /// The origin to use for oriented BIFs
  OriginType m_Origin;

  /// Coefficient to flip the orientation horizontally
  RealType flipHorizontally;
  /// Coefficient to flip the orientation vertically
  RealType flipVertically;

  
  RealImagePointer m_OrientationInX; ///< Set the local reference orientation image in 'x'
  RealImagePointer m_OrientationInY; ///< Set the local reference orientation image in 'y'

  /// Optional mask image
  MaskImagePointer m_Mask;
  MaskFilterPointer m_MaskFilter;

  RealImagePointer m_S11;	///< Second derivative in 'xy'
  RealImagePointer m_S20;	///< Second derivative in 'xx'
  RealImagePointer m_S02;	///< Second derivative in 'yy'

  RealImagePointer m_Orientation;	///< Orientation of the structure

private:
  LightLineResponseImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkLightLineResponseImageFilter.txx"
#endif

#endif
