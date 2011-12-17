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
	typedef MIDASRegionGrowingImageFilter Self;
	typedef SmartPointer<const Self> ConstPointer;
	typedef SmartPointer<Self> Pointer;
	typedef TInputImage InputImageType;
	typedef TOutputImage OutputImageType;
	typedef typename OutputImageType::Pointer OutputImagePointerType;
	typedef typename OutputImageType::ConstPointer OutputImageConstPointerType;
	typedef TPointSet PointSetType;
	typedef typename InputImageType::PixelType InputPixelType;
	typedef typename OutputImageType::PixelType OutputPixelType;
	typedef ImageToImageFilter<TInputImage, TOutputImage> Superclass;
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
	InputPixelType m_LowerThreshold, m_UpperThreshold;
	OutputPixelType m_ForegroundValue, m_BackgroundValue;
	typename PointSetType::ConstPointer mspc_SeedPoints;
	typename OutputImageType::ConstPointer m_ContourImage;

public:
	itkGetConstMacro(LowerThreshold, InputPixelType);
	itkGetConstMacro(UpperThreshold, InputPixelType);
	itkSetMacro(LowerThreshold, InputPixelType);
	itkSetMacro(UpperThreshold, InputPixelType);

	itkGetConstMacro(ForegroundValue, OutputPixelType);
	itkGetConstMacro(BackgroundValue, OutputPixelType);
	itkSetMacro(ForegroundValue, OutputPixelType);
	itkSetMacro(BackgroundValue, OutputPixelType);

	const PointSetType& GetSeedPoints(void) const {
		return *mspc_SeedPoints;
	}

	void SetSeedPoints(const PointSetType &seeds) {
		mspc_SeedPoints = &seeds;
		this->Modified();
	}

	const OutputImageType* GetContourImage(void) const {
		return m_ContourImage;
	}

	itkSetObjectMacro(ContourImage, OutputImageType);
	/** @} */

	/**
	 * \name Region Growing Implementation
	 * 	@{
	 */
private:
	void ConditionalAddPixel(std::stack<typename OutputImageType::IndexType> &r_stack, const typename OutputImageType::IndexType &imgIdx);

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
	MIDASRegionGrowingImageFilter(void) {
		this->SetNumberOfThreads(0);
	}

	virtual ~MIDASRegionGrowingImageFilter(void) {}
	/** @} */
};

#include "itkMIDASRegionGrowingImageFilter.txx"
}
#endif /* _itkMIDASRegionGrowingImageFilter_h_ */
