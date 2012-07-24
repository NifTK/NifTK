/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: $
 Revision          : $Revision: $
 Last modified by  : $Author: ad $

 Original author   : a.duttaroy@cs.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef itkMIDASMeanIntensityWithinARegionFilter_h
#define itkMIDASMeanIntensityWithinARegionFilter_h

#include "itkImageToImageFilter.h"

namespace itk
{

/**
 * \class MIDASMeanIntensityWithinARegionFilter
 * \brief calculate the mean intensity within a binary mask (region).
 *
 * MIDASMeanIntensityWithinARegionFilter has two input images and one 
 * output image. The first input is specified using SetGreyScaleImageInput(),
 * and should be grey scale, and the second input is specified using
 * SetBinaryImageInput and should be a binary mask. The output is
 * a copy of the input binary mask.
 * 
 * \ingroup midas_morph_editor
 */

  template <class TInputImage1, class TInputImage2, class TOutputImage>
  class ITK_EXPORT MIDASMeanIntensityWithinARegionFilter : public ImageToImageFilter<TInputImage1, TOutputImage>
  {
  public:
    /** Standard class typedefs */
    typedef MIDASMeanIntensityWithinARegionFilter           Self;
    typedef ImageToImageFilter<TInputImage1, TOutputImage>  SuperClass;
    typedef SmartPointer<Self>                              Pointer;
    typedef SmartPointer<const Self>                        ConstPointer;

    /** Method for creation through the object factory */
    itkNewMacro(Self);

    /** Run-time type information (and related methods) */
    itkTypeMacro(MIDASMeanIntensityWithinARegionFilter, ImageToImageFilter);

    /** Some additional typedefs */
    typedef TInputImage1                              InputMainImageType;
    typedef typename InputMainImageType::ConstPointer InputMainImageConstPointer;
    typedef typename InputMainImageType::Pointer      InputMainImagePointer;
    typedef typename InputMainImageType::RegionType   InputMainImageRegionType;
    typedef typename InputMainImageType::PixelType    InputMainImagePixelType;
    typedef typename InputMainImageType::IndexType    InputMainImageIndexType;
    typedef typename InputMainImageType::SizeType     InputMainImageSizeType;

    typedef TInputImage2                              InputMaskImageType;
    typedef typename InputMaskImageType::ConstPointer InputMaskImageConstPointer;
    typedef typename InputMaskImageType::Pointer      InputMaskImagePointer;
    typedef typename InputMaskImageType::RegionType   InputMaskImageRegionType;
    typedef typename InputMaskImageType::PixelType    InputMaskImagePixelType;
    typedef typename InputMaskImageType::IndexType    InputMaskImageIndexType;
    typedef typename InputMaskImageType::SizeType     InputMaskImageSizeType;

    typedef TInputImage1                              OutputImageType;
    typedef typename OutputImageType::Pointer         OutputImagePointer;
    typedef typename OutputImageType::RegionType      OutputImageRegionType;
    typedef typename OutputImageType::SizeType        OutputImageSizeType;
    typedef typename OutputImageType::IndexType       OutputImageIndexType;

    /** Set the first input, for the grey scale image. */
    void SetGreyScaleImageInput(const InputMainImageType* image);

    /** Set the second input, which is the binary mask, that will be eroded/dilated. */
    void SetBinaryImageInput(const InputMaskImageType* image);

    /** Method that retrieves the region mean, which is only valid after a successful Update() has been called. */
    double GetMeanIntensityMainImage();

    /** Set/Get the mask value that is considered 'inside' the region. Masks are normally [0,1], or [0,255], where 1 or 255 are considered within the region. */
    itkSetMacro(InValue, InputMaskImagePixelType);
    itkGetConstMacro(InValue, InputMaskImagePixelType);

  protected:
    MIDASMeanIntensityWithinARegionFilter();
    virtual ~MIDASMeanIntensityWithinARegionFilter() {};
    void PrintSelf(std::ostream& os, Indent indent) const;

    /** Pass the input through unmodified. Do this by Grafting in the AllocateOutputs method. */
    void AllocateOutputs();

    /** Do all initialization and other general stuff before starting the threads */
    virtual void BeforeThreadedGenerateData();
    
    // The main method to implement in derived classes, note, its threaded.
    virtual void ThreadedGenerateData(const InputMainImageRegionType &outputRegionForThread, int ThreadID);

    /** Do all the final calculations and other general stuff after the threads finish executing */
    virtual void AfterThreadedGenerateData();

  private:
    MIDASMeanIntensityWithinARegionFilter(const Self&); //purposely not implemented
    void operator=(const Self&); //purposely not implemented
   
    double                m_MeanIntensityMainImage;
    std::vector<double>   m_TotalIntensityVector;
    std::vector<int>      m_CountPixelsVector;
    InputMaskImagePixelType m_InValue;

  };

} //end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMIDASMeanIntensityWithinARegionFilter.txx"
#endif

#endif
