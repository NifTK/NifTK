/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkSetBoundaryVoxelsToValueFilter_h
#define __itkSetBoundaryVoxelsToValueFilter_h

#include "itkImageToImageFilter.h"

namespace itk {
  
/** \class SetBoundaryVoxelsToValueFilter
 * \brief Image filter class to set all voxels which are on the boundary of the image to
 * a user specified value (or zero by default).
 *
 */

template<class TInputImage, class TOutputImage>
class ITK_EXPORT SetBoundaryVoxelsToValueFilter:
    public ImageToImageFilter< TInputImage, TOutputImage >
{
public:
  /** Standard class typedefs. */
  typedef SetBoundaryVoxelsToValueFilter                       Self;
  typedef ImageToImageFilter< TInputImage,TOutputImage > Superclass;
  typedef SmartPointer< Self >                           Pointer;
  typedef SmartPointer< const Self >                     ConstPointer;
  
  /** Run-time type information (and related methods).   */
  itkTypeMacro( SetBoundaryVoxelsToValueFilter, ImageToImageFilter );

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

  /** Type of the output image */
  typedef TOutputImage                          OutputImageType;
  typedef typename OutputImageType::Pointer     OutputImagePointer;
  typedef typename OutputImageType::RegionType  OutputImageRegionType;
  typedef typename OutputImageType::PixelType   OutputImagePixelType;
  typedef typename OutputImageType::IndexType   OutputImageIndexType;
  typedef typename OutputImageType::PointType   OutputImagePointType;
  typedef typename OutputImageType::SizeType    OutputImageSizeType;

  // Set the value to set voxels on the boundary to
  itkSetMacro( Value, OutputImagePixelType );
  // Get the value to set voxels on the boundary to
  itkGetMacro( Value, OutputImagePixelType );

#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro(DimensionShouldBe3,
		  (Concept::SameDimension<itkGetStaticConstMacro(InputImageDimension),3>));
  itkConceptMacro(InputHasNumericTraitsCheck,
                  (Concept::HasNumericTraits<InputImagePixelType>));
  itkConceptMacro(OutputHasPixelTraitsCheck,
                  (Concept::HasPixelTraits<OutputImagePixelType>));
  /** End concept checking */
#endif

protected:
  SetBoundaryVoxelsToValueFilter();
  virtual ~SetBoundaryVoxelsToValueFilter() {};
  void PrintSelf(std::ostream& os, Indent indent) const;
  
  /// Execute the filter
  void GenerateData( );

  // The value to set voxels on the boundary to
  OutputImagePixelType m_Value;

private:
  SetBoundaryVoxelsToValueFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkSetBoundaryVoxelsToValueFilter.txx"
#endif

#endif
