/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
 Dementia Research Centre, and the Centre For Medical Image Computing
 at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
 http://cmic.cs.ucl.ac.uk/
 http://www.ucl.ac.uk/

 Last Changed      : $LastChangedDate: 2011-10-06 10:55:39 +0100 (Thu, 06 Oct 2011) $
 Revision          : $LastChangedRevision: 7447 $
 Last modified by  : $LastChangedBy: mjc $

 Original author   : stian.johnsen.09@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef _itkMIDASRegionGrowingImageFilter_h_
#define _itkMIDASRegionGrowingImageFilter_h_

#include <stack>
#include <cassert>
#include <itkImage.h>
#include <itkSpatialObjectToImageFilter.h>
#include <itkImageToImageFilter.h>
#include <itkImageFileWriter.h>
#include <itkBinaryThresholdImageFilter.h>
#include <itkBinaryFunctorImageFilter.h>

namespace itk {
template <class TInputImage, class TOutputImage, class TPointSet>
class ITK_EXPORT MIDASRegionGrowingImageFilter : public ImageToImageFilter<TInputImage, TOutputImage> {
	/**
	 * \name Standard ITK Types
	 * 	@{
	 */
public:
	typedef MIDASRegionGrowingImageFilter                 Self;
	typedef SmartPointer<const Self>                      ConstPointer;
	typedef SmartPointer<Self>                            Pointer;
	typedef ImageToImageFilter<TInputImage, TOutputImage> Superclass;
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

	/** @} */

	/**
	 * \name ITK image filter standard functions
	 * @{
	 */
public:
	itkNewMacro(Self);
	/** @} */

	/**
	 * \name Region Growing Parameters
	 * @{
	 */
private:
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
	bool                                   m_EraseFullSlice;
	OutputImageIndexType                   m_PropMask;

public:

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

	const PointSetType& GetSeedPoints(void) const {
		return *mspc_SeedPoints;
	}

	void SetSeedPoints(const PointSetType &seeds) {
		mspc_SeedPoints = &seeds;
		this->Modified();
	}

	const OutputImageType* GetSegmentationContourImage(void) const {
		return m_SegmentationContourImage;
	}

	itkSetObjectMacro(SegmentationContourImage, OutputImageType);

  const OutputImageType* GetManualContourImage(void) const {
    return m_ManualContourImage;
  }

  itkSetObjectMacro(ManualContourImage, OutputImageType);

	/** @} */

	/**
	 * \name Region Growing Implementation
	 * 	@{
	 */
private:
	void ConditionalAddPixel(
	    std::stack<typename OutputImageType::IndexType> &r_stack,
	    const typename OutputImageType::IndexType &currentImgIdx,
	    const typename OutputImageType::IndexType &nextImgIdx
	    );

protected:
	virtual void GenerateData(void);

	virtual void ThreadedGenerateData(const typename OutputImageType::RegionType &outputRegionForThread, int threadId) {
		std::cerr << "Not supported.\n";
		abort();
	}
	/** @} */

	/**
	 * \name Construction/Destruction
	 * @{
	 */
public:
	MIDASRegionGrowingImageFilter();
	virtual ~MIDASRegionGrowingImageFilter(void) {}
	/** @} */
};

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMIDASRegionGrowingImageFilter.txx"
#endif
}
#endif /* _itkMIDASRegionGrowingImageFilter_h_ */
