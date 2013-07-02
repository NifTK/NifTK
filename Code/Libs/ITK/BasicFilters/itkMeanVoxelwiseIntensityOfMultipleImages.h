/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkMeanVoxelwiseIntensityOfMultipleImages_h
#define __itkMeanVoxelwiseIntensityOfMultipleImages_h

#include <itkImageToImageFilter.h>
#include <itkEulerAffineTransform.h>


namespace itk {
  
/** \class MeanVoxelwiseIntensityOfMultipleImages 
 * \brief Image filter class to calculate the mean image on a voxel by
 * voxel basis of multiple input images.
 *
 */

template<class TInputImage, class TOutputImage>
class ITK_EXPORT MeanVoxelwiseIntensityOfMultipleImages:
    public ImageToImageFilter< TInputImage, TOutputImage >
{
public:
  /** Standard class typedefs. */
  typedef MeanVoxelwiseIntensityOfMultipleImages         Self;
  typedef ImageToImageFilter< TInputImage,TOutputImage > Superclass;
  typedef SmartPointer< Self >                           Pointer;
  typedef SmartPointer< const Self >                     ConstPointer;
  
  /** Run-time type information (and related methods).   */
  itkTypeMacro( MeanVoxelwiseIntensityOfMultipleImages, ImageToImageFilter );

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
  typedef typename InputImageType::IndexType    InputImageIndexType;
  typedef typename InputImageType::PointType    InputImagePointType;

  typedef typename InputImageRegionType::SizeType    InputImageSizeType;

  /** Type of the output image */
  typedef TOutputImage                          OutputImageType;
  typedef typename OutputImageType::Pointer     OutputImagePointer;
  typedef typename OutputImageType::RegionType  OutputImageRegionType;
  typedef typename OutputImageType::PixelType   OutputImagePixelType;
  typedef typename OutputImageType::SpacingType OutputImageSpacingType;
  typedef typename OutputImageType::IndexType   OutputImageIndexType;
  typedef typename OutputImageType::PointType   OutputImagePointType;

  typedef typename OutputImageRegionType::SizeType    OutputImageSizeType;

  typedef typename itk::EulerAffineTransform< double, ImageDimension, ImageDimension > TransformType;

  typedef typename TransformType::TranslationType TranslationType;
  typedef typename TransformType::InputPointType CenterType;
  typedef typename TransformType::ScaleType ScaleType;

  itkSetMacro( SubtractMinima, bool );
  itkGetMacro( SubtractMinima, bool );

  itkSetMacro( ExpandOutputRegion, double );
  itkGetMacro( ExpandOutputRegion, double );

  /** Set the individual image translations */
  void SetTranslations( std::vector< TranslationType > &translations ) {
    m_TranslationVectors = translations;
    this->Modified();
  }
  /** Set the individual image centers */
  void SetCenters( std::vector< CenterType > &centers ) {
    m_CenterVectors = centers;
    this->Modified();
  }
  /** Set the individual image scales */
  void SetScales( std::vector< ScaleType > &scales ) {
    m_ScaleVectors = scales;
    this->Modified();
  }
  /** Clear the list of transformation parameters */
  void ClearTransformation( void ) {
    m_TranslationVectors.clear();
    m_CenterVectors.clear();
    m_ScaleVectors.clear();
    this->Modified();
  }

#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro(InputHasNumericTraitsCheck,
                  (Concept::HasNumericTraits<InputImagePixelType>));
  itkConceptMacro(OutputHasPixelTraitsCheck,
                  (Concept::HasPixelTraits<OutputImagePixelType>));
  /** End concept checking */
#endif

protected:
  MeanVoxelwiseIntensityOfMultipleImages();
  virtual ~MeanVoxelwiseIntensityOfMultipleImages() {};
  void PrintSelf(std::ostream& os, Indent indent) const;
  
  void GenerateOutputInformation();
  virtual void GenerateInputRequestedRegion();

  void GenerateData();

  /// An optional array of translation vectors
  std::vector< TranslationType > m_TranslationVectors;
  /// An optional array of centers vectors
  std::vector< CenterType > m_CenterVectors;
  /// An optional array of scale vectors
  std::vector< ScaleType > m_ScaleVectors;

  // Create a border around the mean image
  double m_ExpandOutputRegion;

  bool m_SubtractMinima;

  OutputImageRegionType  m_OutRegion;
  OutputImageSizeType    m_OutSize;
  OutputImageSpacingType m_OutSpacing;
  OutputImagePointType   m_OutOrigin;


private:
  MeanVoxelwiseIntensityOfMultipleImages(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMeanVoxelwiseIntensityOfMultipleImages.txx"
#endif

#endif
