/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkRescaleImageUsingHistogramPercentilesFilter_h
#define itkRescaleImageUsingHistogramPercentilesFilter_h

#include <itkImageToImageFilter.h>


namespace itk {
  
/** \class RescaleImageUsingHistogramPercentilesFilter
 * \brief Filter to rescale an image, with the input limits being specified as percentiles of the input image histogram.
 *
 */

template<class TInputImage, class TOutputImage>
class ITK_EXPORT RescaleImageUsingHistogramPercentilesFilter:
    public ImageToImageFilter< TInputImage, TOutputImage >
{
public:
  /** Standard class typedefs. */
  typedef RescaleImageUsingHistogramPercentilesFilter    Self;
  typedef ImageToImageFilter< TInputImage,TOutputImage > Superclass;
  typedef SmartPointer< Self >                           Pointer;
  typedef SmartPointer< const Self >                     ConstPointer;
  
  /** Run-time type information (and related methods).   */
  itkTypeMacro( RescaleImageUsingHistogramPercentilesFilter, ImageToImageFilter );

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

  typedef typename NumericTraits<InputImagePixelType>::RealType RealType;

  /** Type of the output image */
  typedef TOutputImage                          OutputImageType;
  typedef typename OutputImageType::Pointer     OutputImagePointer;
  typedef typename OutputImageType::RegionType  OutputImageRegionType;
  typedef typename OutputImageType::PixelType   OutputImagePixelType;
  typedef typename OutputImageType::IndexType   OutputImageIndexType;
  typedef typename OutputImageType::PointType   OutputImagePointType;

  typedef OutputImagePointType OriginType;

  /// Set the debugging output
  void SetDebug(bool b) { itk::Object::SetDebug(b); }
  /// Set debugging output on
  void DebugOn() { this->SetDebug(true); }
  /// Set debugging output off
  void DebugOff() { this->SetDebug(false); }

  /// Set the verbose output
  void SetVerbose(bool b) { m_FlgVerbose = b; }
  /// Set verbose output on
  void VerboseOn() { this->SetVerbose(true); }
  /// Set verbose output off
  void VerboseOff() { this->SetVerbose(false); }

  /// Clip the output image to the output limits
  void ClipTheOutput() { m_FlgClipTheOutput = true; }

  /// Set the input image lower percentile limit
  itkSetMacro( InLowerPercentile, RealType );
  /// Get the input image lower percentile limit
  itkGetMacro( InLowerPercentile, RealType );

  /// Set the input image upper percentile limit
  itkSetMacro( InUpperPercentile, RealType );
  /// Get the input image upper percentile limit
  itkGetMacro( InUpperPercentile, RealType );

  /// Set the output image lower limit
  itkSetMacro( OutLowerLimit, RealType );
  /// Get the output image lower limit
  itkGetMacro( OutLowerLimit, RealType );

  /// Set the output image upper limit
  itkSetMacro( OutUpperLimit, RealType );
  /// Get the output image upper limit
  itkGetMacro( OutUpperLimit, RealType );


#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro(InputHasNumericTraitsCheck,
                  (Concept::HasNumericTraits<InputImagePixelType>));
  itkConceptMacro(OutputHasPixelTraitsCheck,
                  (Concept::HasPixelTraits<OutputImagePixelType>));
  /** End concept checking */
#endif


protected:
  RescaleImageUsingHistogramPercentilesFilter();
  virtual ~RescaleImageUsingHistogramPercentilesFilter();
  void PrintSelf(std::ostream& os, Indent indent) const;

  /// Rescale the input image
  virtual void GenerateData(void);
  
  /// Flag indicating verbsoe output
  bool m_FlgVerbose;

  /// Clip the output image to the output limits
  bool m_FlgClipTheOutput;

  /// The lower percentile for the input image range
  RealType m_InLowerPercentile;
  /// The upper percentile for the input image range
  RealType m_InUpperPercentile;

  /// The lower limit for the output image range
  RealType m_OutLowerLimit;
  /// The upper limit for the output image range
  RealType m_OutUpperLimit;



private:
  RescaleImageUsingHistogramPercentilesFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkRescaleImageUsingHistogramPercentilesFilter.txx"
#endif

#endif
