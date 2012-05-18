/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-10-06 10:55:39 +0100 (Thu, 06 Oct 2011) $
 Revision          : $Revision: 7447 $
 Last modified by  : $Author: mjc $

 Original author   : a.duttaroy@cs.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef itkMIDASThresholdAndAxialCutoffFilter_h
#define itkMIDASThresholdAndAxialCutoffFilter_h

#include "itkImageToImageFilter.h"
#include <map>
#include <algorithm>


namespace itk
{

/**
 * \class MIDASThresholdAndAxialCutoffFilter
 * \brief Performs a thresholding, and also given a region of interest will blank outside that region.
 *
 * MIDASThresholdAndAxialCutoffFilter has one input image and one 
 * output image. 
 * 
 */

  template <class TImage>
  class ITK_EXPORT MIDASThresholdAndAxialCutoffFilter : public ImageToImageFilter<TImage, TImage>
  {
  public:
    /** Standard class typedefs */
    typedef MIDASThresholdAndAxialCutoffFilter Self;
    typedef ImageToImageFilter<TImage, TImage> SuperClass;
    typedef SmartPointer<Self>                 Pointer;
    typedef SmartPointer<const Self>           ConstPointer;

    /** Typedef to describe the type of pixel. */
    typedef typename TImage::PixelType PixelType;

    /** Typedef to describe the type of pixel index. */
    typedef typename TImage::IndexType IndexType;

    /** Typedef to describe the type of region. */
    typedef typename TImage::RegionType RegionType;

    /** Typedef to describe the size of image region. */
    typedef typename TImage::SizeType SizeType;

    /** Method for creation through the object factory */
    itkNewMacro(Self);

    /** Run-time type information (and related methods) */
    itkTypeMacro(MIDASThresholdAndAxialCutoffFilter, ImageToImageFilter);

    /** Some additional typedefs */
    typedef TImage                                 InputImageType;
    typedef typename InputImageType::ConstPointer  InputImageConstPointer;
    typedef typename InputImageType::RegionType    InputImageRegionType;

    typedef TImage                                 OutputImageType;
    typedef typename OutputImageType::Pointer      OutputImagePointer;
    typedef typename OutputImageType::RegionType   OutputImageRegionType;


    /** Set/Get methods to set the lower threshold */
    itkSetMacro(LowerThreshold, PixelType);
    itkGetConstMacro(LowerThreshold, PixelType);

    /** Set/Get methods to set the upper threshold */
    itkSetMacro(UpperThreshold, PixelType);
    itkGetConstMacro(UpperThreshold, PixelType);

    /** Set/Get methods to set the input value */
    itkSetMacro(InsideRegionValue, PixelType);
    itkGetConstMacro(InsideRegionValue, PixelType);

    /** Set/Get methods to set the output value */
    itkSetMacro(OutsideRegionValue, PixelType);
    itkGetConstMacro(OutsideRegionValue, PixelType);

    /** Set/Get methods to set the region */
    itkSetMacro(RegionToProcess, RegionType);
    itkGetConstMacro(RegionToProcess, RegionType);

    /** Set/Get methods to set whether to process the region or not */
    itkSetMacro(UseRegionToProcess, bool);
    itkGetConstMacro(UseRegionToProcess, bool);


  protected:
    MIDASThresholdAndAxialCutoffFilter();
    virtual ~MIDASThresholdAndAxialCutoffFilter() {};
    void PrintSelf(std::ostream& os, Indent indent) const;

    /** Do all initialization and other general stuff before starting the threads */
    virtual void BeforeThreadedGenerateData();
    
    // The main method to implement in derived classes, note, its threaded.
    virtual void ThreadedGenerateData(const InputImageRegionType &outputRegionForThread, int ThreadID);

  private:
    MIDASThresholdAndAxialCutoffFilter(const Self&); //purposely not implemented
    void operator=(const Self&); //purposely not implemented
   
    bool         m_UseRegionToProcess;
    PixelType    m_LowerThreshold;
    PixelType    m_UpperThreshold;
    PixelType    m_InsideRegionValue;
    PixelType    m_OutsideRegionValue;
    RegionType   m_RegionToProcess;

  };

} //end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMIDASThresholdAndAxialCutoffFilter.txx"
#endif

#endif
