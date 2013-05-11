/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkForwardProjectionWithAffineTransformDifferenceFilter_h
#define __itkForwardProjectionWithAffineTransformDifferenceFilter_h

#include <itkImageToImageFilter.h>
#include <itkConceptChecking.h>

#include <itkCreateForwardBackwardProjectionMatrix.h>
#include "itkBackwardImageProjector2Dto3D.h"

#include <itkPerspectiveProjectionTransform.h>
#include <itkEulerAffineTransform.h>
#include "itkProjectionGeometry.h"
#include "itkSubtract2DImageFromVolumeSliceFilter.h"

namespace itk
{
  
/** \class ForwardProjectionWithAffineTransformDifferenceFilter
 * \brief Class to compute the difference between a reconstruction
 * estimate and the target set of 2D projection images.
 * 
 * This class performs a forward projection of the input volume,
 * calculates the difference between this projection and the
 * corresponding slice of the input volume of 2D projection images and
 * back-projects these differences into the output 3D volume. This
 * process is repeated for each slice of the input volume of 2D
 * projection images and the back-projected differences are summed.
 */

template <class IntensityType = float>
class ITK_EXPORT ForwardProjectionWithAffineTransformDifferenceFilter : 
  public ImageToImageFilter<Image<IntensityType, 3>,
			    Image<IntensityType, 3> > 
{
public:

  /** Standard class typedefs. */
  typedef ForwardProjectionWithAffineTransformDifferenceFilter      Self;
  typedef ImageToImageFilter<Image< IntensityType, 3>, 
			     Image< IntensityType, 3> > Superclass;
  typedef SmartPointer<Self>                            						Pointer;
  typedef SmartPointer<const Self>                      						ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ForwardProjectionWithAffineTransformDifferenceFilter, ImageToImageFilter);

  /** Some convenient typedefs. */
  typedef Image<IntensityType, 3>                   InputVolumeType;
  typedef typename    InputVolumeType::Pointer      InputVolumePointer;
  typedef typename    InputVolumeType::ConstPointer InputVolumeConstPointer;
  typedef typename    InputVolumeType::RegionType   InputVolumeRegionType;
  typedef typename    InputVolumeType::PixelType    InputVolumePixelType;
  typedef typename    InputVolumeType::SizeType     InputVolumeSizeType;
  typedef typename    InputVolumeType::SpacingType  InputVolumeSpacingType;
  typedef typename    InputVolumeType::PointType    InputVolumePointType;

  typedef Image<IntensityType, 3>                             InputProjectionVolumeType;
  typedef typename    InputProjectionVolumeType::Pointer      InputProjectionVolumePointer;
  typedef typename    InputProjectionVolumeType::RegionType   InputProjectionVolumeRegionType;
  typedef typename    InputProjectionVolumeType::PixelType    InputProjectionVolumePixelType;
  typedef typename    InputProjectionVolumeType::SizeType     InputProjectionVolumeSizeType;
  typedef typename    InputProjectionVolumeType::SpacingType  InputProjectionVolumeSpacingType;
  typedef typename    InputProjectionVolumeType::PointType    InputProjectionVolumePointType;

  typedef Image<IntensityType, 3>                                      OutputBackProjectedDifferencesType;
  typedef typename     OutputBackProjectedDifferencesType::Pointer     OutputBackProjectedDifferencesPointer;
  typedef typename     OutputBackProjectedDifferencesType::RegionType  OutputBackProjectedDifferencesRegionType;
  typedef typename     OutputBackProjectedDifferencesType::PixelType   OutputBackProjectedDifferencesPixelType;
  typedef typename     OutputBackProjectedDifferencesType::SizeType    OutputBackProjectedDifferencesSizeType;
  typedef typename     OutputBackProjectedDifferencesType::SpacingType OutputBackProjectedDifferencesSpacingType;
  typedef typename     OutputBackProjectedDifferencesType::PointType   OutputBackProjectedDifferencesPointType;

  typedef itk::EulerAffineTransform<double, 3, 3> EulerAffineTransformType;
  typedef typename EulerAffineTransformType::Pointer EulerAffineTransformPointer;

  typedef itk::PerspectiveProjectionTransform<double> PerspectiveProjectionTransformType;
  typedef typename PerspectiveProjectionTransformType::Pointer PerspectiveProjectionTransformPointer;

  typedef itk::CreateForwardBackwardProjectionMatrix<IntensityType> CreateForwardBackwardProjectionMatrixType;
  typedef typename CreateForwardBackwardProjectionMatrixType::Pointer CreateForwardBackwardProjectionMatrixPointer;

  typedef typename CreateForwardBackwardProjectionMatrixType::OutputImageType ForwardProjectorOutputImageType;
  typedef typename ForwardProjectorOutputImageType::Pointer         ForwardProjectorOutputImagePointer;
  typedef typename ForwardProjectorOutputImageType::RegionType      ForwardProjectorOutputImageRegionType;
  typedef typename ForwardProjectorOutputImageType::SizeType        ForwardProjectorOutputImageSizeType;
  typedef typename ForwardProjectorOutputImageType::SpacingType     ForwardProjectorOutputImageSpacingType;
  typedef typename ForwardProjectorOutputImageType::PointType       ForwardProjectorOutputImagePointType;
  typedef typename ForwardProjectorOutputImageType::PixelType       ForwardProjectorOutputImagePixelType;
  typedef typename ForwardProjectorOutputImageType::IndexType       ForwardProjectorOutputImageIndexType;

  typedef itk::Subtract2DImageFromVolumeSliceFilter<IntensityType> Subtract2DImageFromVolumeSliceFilterType;
  typedef typename Subtract2DImageFromVolumeSliceFilterType::Pointer Subtract2DImageFromVolumeSliceFilterPointer;

  typedef itk::BackwardImageProjector2Dto3D<IntensityType> BackwardImageProjector2Dto3DType;
  typedef typename BackwardImageProjector2Dto3DType::Pointer BackwardImageProjector2Dto3DPointer;

  typedef itk::ProjectionGeometry<IntensityType> ProjectionGeometryType;
  typedef typename ProjectionGeometryType::Pointer ProjectionGeometryPointer;

  /// Set the 3D reconstruction estimate volume input
  void SetInputVolume( InputVolumeType *im3D);

  /// Set the input 3D volume of projection image set one
  void SetInputProjectionVolumeOne( InputProjectionVolumeType *im2D);
	/// Set the input 3D volume of projection image set two
  void SetInputProjectionVolumeTwo( InputProjectionVolumeType *im2D);

  /// Get/Set the projection geometry
  itkSetObjectMacro( ProjectionGeometry, ProjectionGeometryType );
  itkGetObjectMacro( ProjectionGeometry, ProjectionGeometryType );

  /** Return a pointer to the input volume */
  InputVolumePointer GetPointerToInputVolume(void);

  /** ImageDimension enumeration */
  itkStaticConstMacro(InputVolumeDimension, unsigned int,
                      InputVolumeType::ImageDimension);
  itkStaticConstMacro(OutputBackProjectedDifferencesDimension, unsigned int,
                      OutputBackProjectedDifferencesType::ImageDimension);

  /** Rather than calculate the input requested region for a
   * particular back-projection (which might take longer than the
   * actual projection), we simply set the input requested region to
   * the entire area of the current 3D volume.
   * \sa ProcessObject::GenerateInputRequestedRegion() */
  virtual void GenerateInputRequestedRegion();
  virtual void EnlargeOutputRequestedRegion(DataObject *output); 

  /// Initialise the image pipeline
  void Initialise(void);
  
  /// Set the backprojection volume to zero prior to the next back-projection
  void ClearVolumePriorToNextBackProjection(void) {m_BackProjectorOne->ClearVolumePriorToNextBackProjection();}

protected:

  ForwardProjectionWithAffineTransformDifferenceFilter();
  virtual ~ForwardProjectionWithAffineTransformDifferenceFilter() {};

  void PrintSelf(std::ostream& os, Indent indent) const;

  /// The function called when the filter is executed
  void GenerateData();

  /// Flag indicating whether the filter has been initialised
  bool m_FlagPipelineInitialised;

  /// The number of projection images
  unsigned int m_NumberOfProjections;

  /// The forward-projector one
  CreateForwardBackwardProjectionMatrixPointer m_ForwardProjectorOne;

  /// A filter to perform the subtraction one
  Subtract2DImageFromVolumeSliceFilterPointer m_SubtractProjectionFromEstimateOne;

  /// The back-projector one
  BackwardImageProjector2Dto3DPointer m_BackProjectorOne;

  /// The forward-projector two
  CreateForwardBackwardProjectionMatrixPointer m_ForwardProjectorTwo;

  /// A filter to perform the subtraction two
  Subtract2DImageFromVolumeSliceFilterPointer m_SubtractProjectionFromEstimateTwo;

  /// The back-projector two
  BackwardImageProjector2Dto3DPointer m_BackProjectorTwo;

  /// The specific projection geometry to be used
  ProjectionGeometryPointer m_ProjectionGeometry;


private:
  ForwardProjectionWithAffineTransformDifferenceFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
  
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkForwardProjectionWithAffineTransformDifferenceFilter.txx"
#endif

#endif
