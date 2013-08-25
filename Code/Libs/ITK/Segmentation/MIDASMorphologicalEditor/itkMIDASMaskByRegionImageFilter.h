/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkMIDASMaskByRegionImageFilter_h
#define itkMIDASMaskByRegionImageFilter_h

#include <itkImageToImageFilter.h>

namespace itk
{

  /**
   * \class MIDASMaskByRegionImageFilter
   * \brief Class, developed for MIDAS migration, that outputs an image the same size as the input,
   * but you can specify a region, and that region is kept, and anything outside that region, set
   * to a single background value. Used for Axial Cutoff.
   *
   * Within the region, the spec is:
   * <pre>
   * Input 0  | Input 1 | Input 2 | Output
   *       0  |       0 |       0 |      0
   *       0  |       0 |       1 |      0
   *       0  |       1 |       0 |      1
   *       0  |       1 |       1 |      0
   *       1  |       0 |       0 |      1
   *       1  |       0 |       1 |      0
   *       1  |       1 |       0 |      1
   *       1  |       1 |       1 |      0
   * </pre>
   * and input 1 has been called the "additions" image, as if the pixel is on, it has the effect of
   * adding to the segmentation. The second image is called the "subtractions" image, and is used for
   * connection breaker, so if the subtractions image pixel is on, the output must be off.
   *
   * \ingroup midas_morph_editor
   */
  template <class TInputImage, class TOutputImage>
  class ITK_EXPORT MIDASMaskByRegionImageFilter : public ImageToImageFilter<TInputImage, TOutputImage>
  {

    public:

      /** Standard class typedefs */
      typedef MIDASMaskByRegionImageFilter                          Self;
      typedef ImageToImageFilter<TInputImage, TOutputImage>         SuperClass;
      typedef SmartPointer<Self>                                    Pointer;
      typedef SmartPointer<const Self>                              ConstPointer;

      /** Method for creation through the object factory */
      itkNewMacro(Self);

      /** Run-time type information (and related methods) */
      itkTypeMacro(MIDASMaskByRegionImageFilter, ImageToImageFilter);

      /** Standard Typedefs. */
      typedef typename TInputImage::IndexType                       IndexType;
      typedef typename TInputImage::SizeType                        SizeType;
      typedef typename TInputImage::RegionType                      RegionType;
      typedef typename TInputImage::PixelType                       InputPixelType;
      typedef typename TOutputImage::PixelType                      OutputPixelType;

      /** Set/Get methods to set the region to keep. */
      void SetRegion(RegionType region) { m_Region = region; m_UserSetRegion = true; this->Modified(); }
      RegionType GetRegion() const { return m_Region; }

      /** Set/Get methods to set the output background value. Default 0. */
      itkSetMacro(OutputBackgroundValue, OutputPixelType);
      itkGetConstMacro(OutputBackgroundValue, OutputPixelType);

      /** Set/Get methods to set the flag controlling whether we actually use the region or not. Defaults false. */
      itkSetMacro(UserSetRegion, bool);
      itkGetConstMacro(UserSetRegion, bool);

    protected:

      MIDASMaskByRegionImageFilter();
      virtual ~MIDASMaskByRegionImageFilter() {};
      void PrintSelf(std::ostream& os, Indent indent) const;

      virtual void BeforeThreadedGenerateData();
      virtual void ThreadedGenerateData(const RegionType& outputRegionForThread, ThreadIdType threadNumber);

    private:
      MIDASMaskByRegionImageFilter(const Self&); //purposely not implemented
      void operator=(const Self&); //purposely not implemented

      bool m_UserSetRegion;
      RegionType m_Region;
      OutputPixelType m_OutputBackgroundValue;

  }; // end class

} // end namespace

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMIDASMaskByRegionImageFilter.txx"
#endif

#endif
