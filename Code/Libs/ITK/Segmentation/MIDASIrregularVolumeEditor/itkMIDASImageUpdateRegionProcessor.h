/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef ITKMIDASIMAGEUPDATEREGIONPROCESSOR_H
#define ITKMIDASIMAGEUPDATEREGIONPROCESSOR_H

#include "itkMIDASImageUpdateProcessor.h"
#include <itkExtractImageFilter.h>
#include <itkPasteImageFilter.h>

namespace itk
{

/**
 * \class MIDASImageUpdateRegionProcessor
 * \brief Provides methods to do Undo/Redo within a specific Region.
 */
template <class TPixel, unsigned int VImageDimension>
class ITK_EXPORT MIDASImageUpdateRegionProcessor : public MIDASImageUpdateProcessor<TPixel, VImageDimension> {

public:

  /** Standard class typedefs */
  typedef MIDASImageUpdateRegionProcessor                    Self;
  typedef MIDASImageUpdateProcessor<TPixel, VImageDimension> Superclass;
  typedef SmartPointer<Self>                                 Pointer;
  typedef SmartPointer<const Self>                           ConstPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro(MIDASImageUpdateRegionProcessor, MIDASImageUpdateProcessor);

  /** Additional typedefs */
  typedef TPixel PixelType;
  typedef Image<TPixel, VImageDimension>  ImageType;
  typedef typename ImageType::Pointer     ImagePointer;
  typedef typename ImageType::IndexType   IndexType;
  typedef typename ImageType::SizeType    SizeType;
  typedef typename ImageType::RegionType  RegionType;
  typedef itk::ExtractImageFilter<ImageType, ImageType> ExtractImageFilterType;
  typedef typename ExtractImageFilterType::Pointer      ExtractImageFilterPointer;
  typedef itk::PasteImageFilter<ImageType, ImageType>   PasteImageFilterType;
  typedef typename PasteImageFilterType::Pointer        PasteImageFilterPointer;

  /** Set the destination region of interest, which controls the region that is copied into m_BeforeImage and m_After image for Undo/Redo purposes. */
  itkSetMacro(DestinationRegionOfInterest, RegionType);
  itkGetMacro(DestinationRegionOfInterest, RegionType);

  /** Overloaded method to provide simple acess via a std::vector, where we assume the length is 6 corresponding to the first 3 numbers indicating the starting index, and the next 3 numbers indicating the region size. */
  void SetDestinationRegionOfInterest(std::vector<int> &region);

  /** This will copy the m_BeforeImage into the m_DestinationImage */
  virtual void Undo();

  /** This will copy the m_AfterImage into the m_DestinationImage. This method should also be called to execute the whole process first time round. */
  virtual void Redo();

protected:
  MIDASImageUpdateRegionProcessor();
  void PrintSelf(std::ostream& os, Indent indent) const;
  virtual ~MIDASImageUpdateRegionProcessor() {}

  /** Returns the after image, so derived classes can apply an update. */
  itkGetObjectMacro(AfterImage, ImageType);
  itkSetObjectMacro(AfterImage, ImageType);

  /** Derived classes calculate whatever update they like, but can only affect the m_AfterImage, which must be within the destination region of interest. */
  virtual void ApplyUpdateToAfterImage() = 0;

  virtual void ValidateInputs();

private:
  MIDASImageUpdateRegionProcessor(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  void CopyImageRegionToDestination(ImagePointer sourceImage);

  bool         m_UpdateCalculated;
  RegionType   m_DestinationRegionOfInterest;
  ImagePointer m_BeforeImage;
  ImagePointer m_AfterImage;

}; // end class

} // end namespace

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMIDASImageUpdateRegionProcessor.txx"
#endif

#endif // ITKMIDASIMAGEUPDATEREGIONPROCESSOR_H
