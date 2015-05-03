/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkDynamicContrastEnhancementAnalysisImageFilter_h
#define __itkDynamicContrastEnhancementAnalysisImageFilter_h
 
#include <itkImageToImageFilter.h>
 
namespace itk
{
/** \class DynamicContrastEnhancementAnalysisImageFilter
 * \brief Image filter to process a set of contrast enhancement images.
 */

template< class TInputImage, class TOutputImage >
class DynamicContrastEnhancementAnalysisImageFilter : 
  public ImageToImageFilter< TInputImage, TOutputImage >
{
public:
  /** Standard class typedefs. */
  typedef DynamicContrastEnhancementAnalysisImageFilter Self;
  typedef ImageToImageFilter< TInputImage, TOutputImage > Superclass;
  typedef SmartPointer< Self > Pointer;
 
  /** Method for creation through the object factory. */
  itkNewMacro(Self);
 
  /** Run-time type information (and related methods). */
  itkTypeMacro(DynamicContrastEnhancementAnalysisImageFilter, ImageToImageFilter);

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
 
  /** Optional mask image */
  typedef unsigned char                                      MaskPixelType;
  typedef typename itk::Image<MaskPixelType, ImageDimension> MaskImageType;
  typedef typename MaskImageType::Pointer                    MaskImagePointer;


  /** The contrast enhancement image acquired at time 'tAcquired' and
   * number 'iAcquired' in the sequence.*/
  void SetInputImage(const TInputImage *image, RealType tAcquired, unsigned int iAcquired);
 
  /// Set an optional mask image
  itkSetObjectMacro( Mask, MaskImageType );

  /// Get the area under the contrast enhancement curve (subtracted) image
  OutputImageType *GetOutputAUC( void ) {
    return dynamic_cast< OutputImageType* >( this->ProcessObject::GetOutput( 0 ) );
  }

  /// Get the maximum enhancement rate image
  OutputImageType *GetOutputMaxRate( void ) {
    return dynamic_cast< OutputImageType* >( this->ProcessObject::GetOutput( 1 ) );
  }

  /// Get the time to maximum enhancement image
  OutputImageType *GetOutputTime2Max( void ) {
    return dynamic_cast< OutputImageType* >( this->ProcessObject::GetOutput( 2 ) );
  }

  /// Get the maximum enhancement image
  OutputImageType *GetOutputMax( void ) {
    return dynamic_cast< OutputImageType* >( this->ProcessObject::GetOutput( 3 ) );
  }

  /// Get the maximum wash out rate image
  OutputImageType *GetOutputWashOut( void ) {
    return dynamic_cast< OutputImageType* >( this->ProcessObject::GetOutput( 4 ) );
  }

protected:
  DynamicContrastEnhancementAnalysisImageFilter();
  ~DynamicContrastEnhancementAnalysisImageFilter(){}

  /// The number of input images
  unsigned int m_NumberOfInputImages;


  /// Optional mask image
  MaskImagePointer m_Mask;

  std::vector< RealType > m_AcquistionTime;

  /** Does the real work. */
  virtual void GenerateData();

  /**  Create the Output */
  DataObject::Pointer MakeOutput(unsigned int idx);

private:
  DynamicContrastEnhancementAnalysisImageFilter(const Self &); //purposely not implemented
  void operator=(const Self &);  //purposely not implemented
 
};
} //namespace ITK
 
 
#ifndef ITK_MANUAL_INSTANTIATION
#include "itkDynamicContrastEnhancementAnalysisImageFilter.txx"
#endif
 
 
#endif // __itkDynamicContrastEnhancementAnalysisImageFilter_h
