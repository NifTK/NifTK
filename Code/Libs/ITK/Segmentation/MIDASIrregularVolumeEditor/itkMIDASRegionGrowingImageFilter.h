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

	itkSetMacro(LowerThreshold, InputPixelType);
	itkGetConstMacro(LowerThreshold, InputPixelType);

	itkSetMacro(UpperThreshold, InputPixelType);
	itkGetConstMacro(UpperThreshold, InputPixelType);

	itkSetMacro(ForegroundValue, OutputPixelType);
	itkGetConstMacro(ForegroundValue, OutputPixelType);

	itkSetMacro(BackgroundValue, OutputPixelType);
	itkGetConstMacro(BackgroundValue, OutputPixelType);

	itkSetMacro(RegionOfInterest, OutputImageRegionType);
	itkGetConstMacro(RegionOfInterest, OutputImageRegionType);

	itkSetMacro(UseRegionOfInterest, bool);
	itkGetConstMacro(UseRegionOfInterest, bool);

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

  itkSetMacro(EraseFullSlice, bool);
  itkGetConstMacro(EraseFullSlice, bool);

  itkSetMacro(PropMask, OutputImageIndexType);
  itkGetConstMacro(PropMask, OutputImageIndexType);

  itkSetMacro(UsePropMaskMode, bool);
  itkGetConstMacro(UsePropMaskMode, bool);

  void SetManualContours(ParametricPathVectorType* contours);

	const PointSetType& GetSeedPoints(void) const
	{
		return *mspc_SeedPoints;
	}

	void SetSeedPoints(const PointSetType &seeds)
	{
		mspc_SeedPoints = &seeds;
		this->Modified();
	}

	const OutputImageType* GetSegmentationContourImage(void) const
	{
		return m_SegmentationContourImage;
	}

	itkSetObjectMacro(SegmentationContourImage, OutputImageType);

  const OutputImageType* GetManualContourImage(void) const
  {
    return m_ManualContourImage;
  }

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

	void ConditionalAddPixel(
	    std::stack<typename OutputImageType::IndexType> &r_stack,
	    const typename OutputImageType::IndexType &currentImgIdx,
	    const typename OutputImageType::IndexType &nextImgIdx,
	    const bool &isFullyConnected
	    );

	bool IsFullyConnected(
	    const typename OutputImageType::IndexType &index1,
	    const typename OutputImageType::IndexType &index2
	    );

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
