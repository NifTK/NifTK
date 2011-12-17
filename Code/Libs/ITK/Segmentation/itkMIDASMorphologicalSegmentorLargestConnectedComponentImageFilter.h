/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date$
 Revision          : $Rev$
 Last modified by  : $Author$

 Original author   : stian.johnsen.09@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef ITKMIDASMORPHOLOGICALSEGMENTORLARGESTCONNECTEDCOMPONENTIMAGEFILTER_H_
#define ITKMIDASMORPHOLOGICALSEGMENTORLARGESTCONNECTEDCOMPONENTIMAGEFILTER_H_

#include <vector>
#include <stack>
#include <itkImage.h>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>
#include <itkImageToImageFilter.h>

namespace itk {
	/**
	 * \brief Largest component extractor
	 *
	 *
	 * Returns an image only containing the largest of the foreground components of the input image.
	 */
	template <class TInputImageType, class TOutputImageType>
	class MIDASMorphologicalSegmentorLargestConnectedComponentImageFilter : public ImageToImageFilter<TInputImageType, TOutputImageType> {
		/**
		 * \name Types
		 * @{
		 */
	public:
		typedef TInputImageType InputImageType;
		typedef TOutputImageType OutputImageType;
		typedef typename InputImageType::IndexType IndexType;
		typedef typename OutputImageType::PixelType OutputImagePixelType;
		typedef typename InputImageType::PixelType InputImagePixelType;
		typedef typename InputImageType::RegionType InputImageRegionType;
		/** @} */

		/**
		 * \name Output Values
		 * @{
		 */
	private:
		OutputImagePixelType m_OutputBackgroundValue, m_OutputForegroundValue;
		InputImagePixelType m_InputBackgroundValue;
		std::vector<IndexType> m_ComponentIndicies[2];

	public:
	    /** Set/Get methods to set the value on the input image that is considered background. Default 0. */
	    itkSetMacro(InputBackgroundValue, InputImagePixelType);
	    itkGetConstMacro(InputBackgroundValue, InputImagePixelType);

	    /** Set/Get methods to set the output value for outside the largest region. Default 0. */
	    itkSetMacro(OutputBackgroundValue, OutputImagePixelType);
	    itkGetConstMacro(OutputBackgroundValue, OutputImagePixelType);

	    /** Set/Get methods to set the output value for inside the largest region. Default 1. */
	    itkSetMacro(OutputForegroundValue, OutputImagePixelType);
	    itkGetConstMacro(OutputForegroundValue, OutputImagePixelType);

	    /** Set/Get the suggested size of the largest region, to enable vectors to be allocated in one go. Default 1. */
	    void SetCapacity(unsigned long int n);
	    unsigned long int GetCapacity() const;

	    /** @} */

	    /**
	     * \name Processing
	     * @{
	     */
	protected:
	    /** Creates a list of pixel indices belonging to the component starting at "startIndex". Removes the corresponding labels from the output image. */
	    void _ProcessRegion(std::vector<IndexType> &r_regionIndices, const IndexType &startIndex);

	    /** Labels the pixels whose indices are passed in regionIndices */
	    void _SetComponentPixels(const std::vector<IndexType> &regionIndices);

	    /** The main method to implement the connected component labeling in this single-threaded class */
	    virtual void GenerateData();
	    /** @} */

	    /**
	     * \name Standard ITK Filter API
	     * @{
	     */
	public:
	    typedef MIDASMorphologicalSegmentorLargestConnectedComponentImageFilter Self;
	    typedef ImageToImageFilter<InputImageType, OutputImageType> SuperClass;
	    typedef SmartPointer<Self> Pointer;
	    typedef SmartPointer<const Self> ConstPointer;

	public:
	    itkNewMacro(Self);
	    /** @} */

	    /**
	     * \name Construction, Destruction
	     * @{
	     */
	protected:
	    MIDASMorphologicalSegmentorLargestConnectedComponentImageFilter(void);
	    virtual ~MIDASMorphologicalSegmentorLargestConnectedComponentImageFilter(void) {}
	    /** @} */
	};

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMIDASMorphologicalSegmentorLargestConnectedComponentImageFilter.txx"
#endif
}

#endif /* ITKMIDASMORPHOLOGICALSEGMENTORLARGESTCONNECTEDCOMPONENTIMAGEFILTER_H_ */
