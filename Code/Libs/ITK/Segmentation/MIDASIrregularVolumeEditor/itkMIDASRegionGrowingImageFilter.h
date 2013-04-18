/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkMIDASRegionGrowingImageFilter_h
#define itkMIDASRegionGrowingImageFilter_h

#include <stack>
#include <cassert>
#include <itkImage.h>
#include <itkImageToImageFilter.h>
#include <itkImageRegionConstIteratorWithIndex.h>
#include <itkPolyLineParametricPath.h>
#include <itkContinuousIndex.h>

namespace itk {

/**
 * \class MIDASRegionGrowingImageFilter
 * \brief Implements region growing limited by contours.
 */
template <class TInputImage, class TOutputImage, class TPointSet>
class ITK_EXPORT MIDASRegionGrowingImageFilter : public ImageToImageFilter<TInputImage, TOutputImage> {

public:
	typedef MIDASRegionGrowingImageFilter                 Self;
	typedef SmartPointer<const Self>                      ConstPointer;
	typedef SmartPointer<Self>                            Pointer;
	typedef ImageToImageFilter<TInputImage, TOutputImage> Superclass;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(MIDASRegionGrowingImageFilter, ImageToImageFilter );

	typedef TInputImage                                   InputImageType;
	typedef typename InputImageType::PixelType            InputPixelType;
	typedef TOutputImage                                  OutputImageType;
	typedef typename OutputImageType::RegionType          OutputImageRegionType;
	typedef typename OutputImageType::SizeType            OutputImageSizeType;
	typedef typename OutputImageType::IndexType           OutputImageIndexType;
	typedef typename OutputImageType::Pointer             OutputImagePointerType;
	typedef typename OutputImageType::ConstPointer        OutputImageConstPointerType;
	typedef typename OutputImageType::PixelType           OutputPixelType;
	typedef TPointSet                                     PointSetType;

	typedef itk::ContinuousIndex<double,TInputImage::ImageDimension> ContinuousIndexType;
	typedef itk::PolyLineParametricPath<TInputImage::ImageDimension> ParametricPathType;

	typedef typename ParametricPathType::Pointer          ParametricPathPointer;
	typedef std::vector<ParametricPathPointer>            ParametricPathVectorType;
	typedef typename ParametricPathType::VertexListType   ParametricPathVertexListType;
	typedef typename ParametricPathType::VertexType       ParametricPathVertexType;

	/**
	 * \brief Values lower than the LowerThreshold are not grown into.
	 */
	itkSetMacro(LowerThreshold, InputPixelType);
	itkGetConstMacro(LowerThreshold, InputPixelType);

	/**
	 * \brief Values higher than the UpperThreshold are not grown into.
	 */
	itkSetMacro(UpperThreshold, InputPixelType);
	itkGetConstMacro(UpperThreshold, InputPixelType);

	/**
	 * \brief The output "foreground" value, normally 1 or 255.
	 */
	itkSetMacro(ForegroundValue, OutputPixelType);
	itkGetConstMacro(ForegroundValue, OutputPixelType);

	/**
	 * \brief The output "background" value, normally 0.
	 */
	itkSetMacro(BackgroundValue, OutputPixelType);
	itkGetConstMacro(BackgroundValue, OutputPixelType);

	/**
	 * \brief Set a region of interest, and only operate within that.
	 */
	itkSetMacro(RegionOfInterest, OutputImageRegionType);
	itkGetConstMacro(RegionOfInterest, OutputImageRegionType);

	/**
	 * \brief If true, the filter will use RegionOfInterest and otherwise won't.
	 */
	itkSetMacro(UseRegionOfInterest, bool);
	itkGetConstMacro(UseRegionOfInterest, bool);

  /**
   * \brief It may be the case that you need to project seeds into a region.
   * Say you have seeds on one slice that you want to use as seeds for region growing
   * upwards, then you need to shift them along the upwards axis.
   */
  itkSetMacro(ProjectSeedsIntoRegion, bool);
  itkGetConstMacro(ProjectSeedsIntoRegion, bool);

  itkSetMacro(MaximumSeedProjectionDistanceInVoxels, unsigned int);
  itkGetMacro(MaximumSeedProjectionDistanceInVoxels, unsigned int);

  itkSetMacro(SegmentationContourImageInsideValue, OutputPixelType);
  itkGetConstMacro(SegmentationContourImageInsideValue, OutputPixelType);

  itkSetMacro(SegmentationContourImageBorderValue, OutputPixelType);
  itkGetConstMacro(SegmentationContourImageBorderValue, OutputPixelType);

  itkSetMacro(SegmentationContourImageOutsideValue, OutputPixelType);
  itkGetConstMacro(SegmentationContourImageOutsideValue, OutputPixelType);

  itkSetMacro(ManualContourImageBorderValue, OutputPixelType);
  itkGetConstMacro(ManualContourImageBorderValue, OutputPixelType);

  itkSetMacro(ManualContourImageNonBorderValue, OutputPixelType);
  itkGetConstMacro(ManualContourImageNonBorderValue, OutputPixelType);

  /**
   * \brief Within MIDAS, if region growing covers the whole slice, the output
   * is zero, not the whole slice.
   *
   * We need this flag as there are two use cases. When region growing, and we
   * want to see the blue outline of the region growing, then when the whole
   * slice is covered MIDAS erases (or ignores) the region growing, so it disappears.
   * However, for testing whether seeds are enclosed or not, we need to region grow
   * to the edge, and check if we hit it, so we dont want to erase the whole slice.
   */
  itkSetMacro(EraseFullSlice, bool);
  itkGetConstMacro(EraseFullSlice, bool);

  itkSetMacro(PropMask, OutputImageIndexType);
  itkGetConstMacro(PropMask, OutputImageIndexType);

  itkSetMacro(UsePropMaskMode, bool);
  itkGetConstMacro(UsePropMaskMode, bool);

  /**
   * \brief Setting the "manual" contours means those that come from MIDASDrawTool or MIDASPolyTool.
   */
  void SetManualContours(ParametricPathVectorType* contours);

	/**
	 * \brief Retrieve the seeds.
	 */
	const PointSetType& GetSeedPoints(void) const
	{
		return *mspc_SeedPoints;
	}

	/**
	 * \brief Set the seeds, as region growing starts from each seed point.
	 */
	void SetSeedPoints(const PointSetType &seeds)
	{
		mspc_SeedPoints = &seeds;
		this->Modified();
	}

	/**
	 * \brief Retrieve the contour image.
	 */
	const OutputImageType* GetSegmentationContourImage(void) const
	{
		return m_SegmentationContourImage;
	}

	/**
	 * \brief Set the contour image.
	 *
	 * The "contour image" is the image generated by working out the edge of the current segmentation,
	 * and rendering an image of all the borders between foreground and background. The contour image
	 * contains a value for "inside" the contour (i.e. background), the contour itself, and "outside"
	 * which is background that it known to be outside the segmented object.
	 */
	itkSetObjectMacro(SegmentationContourImage, OutputImageType);

  /**
   * \brief Retrieve the "manual" contour image.
   */
  const OutputImageType* GetManualContourImage(void) const
  {
    return m_ManualContourImage;
  }

  /**
   * \brief Set the manual contour image.
   *
   * The manual contour image is the image generated by taking all the lines from
   * MIDASDrawTool and MIDASPolyTool, and rendering them into the image. As these
   * lines may be unclosed (eg. a line, and not a circle), there is no concept of
   * "inside" or "outside" the line.  The only thing we can render is "border"
   * or "not-border".
   */
  itkSetObjectMacro(ManualContourImage, OutputImageType);

protected:

	MIDASRegionGrowingImageFilter(); // purposely hidden
	virtual ~MIDASRegionGrowingImageFilter(void) {} // purposely hidden

	virtual void GenerateData(void);

	virtual void ThreadedGenerateData(const typename OutputImageType::RegionType &outputRegionForThread, int threadId) {
		std::cerr << "Not supported.\n";
		abort();
	}

private:

  MIDASRegionGrowingImageFilter(const Self&); // purposely not implemented
  void operator=(const Self&); // purposely not implemented

	InputPixelType                         m_LowerThreshold;
	InputPixelType                         m_UpperThreshold;
	OutputPixelType                        m_ForegroundValue;
	OutputPixelType                        m_BackgroundValue;
	typename PointSetType::ConstPointer    mspc_SeedPoints;
  OutputImageRegionType                  m_RegionOfInterest;
  bool                                   m_UseRegionOfInterest;
  bool                                   m_ProjectSeedsIntoRegion;
  unsigned int                           m_MaximumSeedProjectionDistanceInVoxels;
	typename OutputImageType::ConstPointer m_SegmentationContourImage;
	OutputPixelType                        m_SegmentationContourImageInsideValue;
	OutputPixelType                        m_SegmentationContourImageBorderValue;
	OutputPixelType                        m_SegmentationContourImageOutsideValue;
	typename OutputImageType::ConstPointer m_ManualContourImage;
	OutputPixelType                        m_ManualContourImageBorderValue;
	OutputPixelType                        m_ManualContourImageNonBorderValue;
	ParametricPathVectorType*              m_ManualContours;
	bool                                   m_EraseFullSlice;
	OutputImageIndexType                   m_PropMask;
	bool                                   m_UsePropMaskMode;

	/**
	 * \brief The main region growing logic is here, where we decide whether to add the nextImgIdx to a stack.
	 * \param[In] r_stack current stack of pixels under consideration, initialised by the available seeds.
	 * \param[In] currentImgIdx the current location being considered
	 * \param[In] nextImgIdx the next pixel
	 * \param[Out] true if the pixel should be added and false otherwise
	 */
	void ConditionalAddPixel(
	    std::stack<typename OutputImageType::IndexType> &r_stack,
	    const typename OutputImageType::IndexType &currentImgIdx,
	    const typename OutputImageType::IndexType &nextImgIdx,
	    const bool &isFullyConnected
	    );

	/**
	 * \brief Will return true if index1 and index2 are joined along an edge rather than a diagonal, and false otherwise.
	 * (Assuming the pixels are next to each other, and not miles apart).
	 */
	bool IsFullyConnected(
	    const typename OutputImageType::IndexType &index1,
	    const typename OutputImageType::IndexType &index2
	    );

	/**
	 * \brief Will return true if index1 and index2 cross a contour line, given by m_ManualContours.
	 */
	bool IsCrossingLine(
	    const typename OutputImageType::IndexType &index1,
	    const typename OutputImageType::IndexType &index2
	    );
};

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMIDASRegionGrowingImageFilter.txx"
#endif

} // end namespace

#endif
