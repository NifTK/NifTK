/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

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
	 * \brief Largest connected component filter.
	 *
	 * Returns an image only containing the largest of the foreground components of the input image.
	 *
	 * \ingroup midas_morph_editor
	 */
	template <class TInputImageType, class TOutputImageType>
	class MIDASMorphologicalSegmentorLargestConnectedComponentImageFilter : public ImageToImageFilter<TInputImageType, TOutputImageType> {

	public:
		typedef TInputImageType InputImageType;
		typedef TOutputImageType OutputImageType;
		typedef typename InputImageType::IndexType IndexType;
		typedef typename OutputImageType::PixelType OutputImagePixelType;
		typedef typename InputImageType::PixelType InputImagePixelType;
		typedef typename InputImageType::RegionType InputImageRegionType;
    typedef MIDASMorphologicalSegmentorLargestConnectedComponentImageFilter Self;
    typedef ImageToImageFilter<InputImageType, OutputImageType> SuperClass;
    typedef SmartPointer<Self> Pointer;
    typedef SmartPointer<const Self> ConstPointer;
    itkNewMacro(Self);

		/** Set/Get methods to set the value on the input image that is considered background. Default 0. */
	  itkSetMacro(InputBackgroundValue, InputImagePixelType);
	  itkGetConstMacro(InputBackgroundValue, InputImagePixelType);

	  /** Set/Get methods to set the output value for outside the largest region. Default 0. */
	  itkSetMacro(OutputBackgroundValue, OutputImagePixelType);
	  itkGetConstMacro(OutputBackgroundValue, OutputImagePixelType);

	  /** Set/Get methods to set the output value for inside the largest region. Default 1. */
	  itkSetMacro(OutputForegroundValue, OutputImagePixelType);
	  itkGetConstMacro(OutputForegroundValue, OutputImagePixelType);

	  /** Set/Get the capcity. */
    itkSetMacro(Capacity, unsigned int);
    itkGetConstMacro(Capacity, unsigned int);

	protected:
    MIDASMorphologicalSegmentorLargestConnectedComponentImageFilter(void);
	  virtual ~MIDASMorphologicalSegmentorLargestConnectedComponentImageFilter(void) {}

	  /** Creates a list of pixel indices belonging to the component starting at "startIndex". Removes the corresponding labels from the output image. */
	  void _ProcessRegion(std::vector<unsigned int> &r_regionIndices, const unsigned int &startIndex);

	  /** Labels the pixels whose indices are passed in regionIndices */
	  void _SetComponentPixels(const std::vector<unsigned int> &regionIndices);

	  /** Initialises stuff before multithreaded section, eg. clearing m_MapOfLabelledPixels. */
	  virtual void BeforeThreadedGenerateData();

	  /** Multi-threaded section, that actually just does some basic initialisation. */
	  virtual void ThreadedGenerateData(const InputImageRegionType &outputRegionForThread, int ThreadID);

	  /** In contrast to conventional ITK style, most of the algorithm is here. */
	  virtual void AfterThreadedGenerateData();

  private:
    OutputImagePixelType           m_OutputBackgroundValue;
    OutputImagePixelType           m_OutputForegroundValue;
    InputImagePixelType            m_InputBackgroundValue;
    unsigned int                   m_Capacity;
    std::vector<unsigned long int> m_NumberOfLabelledPixelsPerThread;
	};

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMIDASMorphologicalSegmentorLargestConnectedComponentImageFilter.txx"
#endif
}

#endif /* ITKMIDASMORPHOLOGICALSEGMENTORLARGESTCONNECTEDCOMPONENTIMAGEFILTER_H_ */
