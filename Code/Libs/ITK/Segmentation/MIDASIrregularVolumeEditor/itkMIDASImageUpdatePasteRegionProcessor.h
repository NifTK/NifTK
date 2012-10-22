/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-10-06 12:21:57 +0100 (Thu, 06 Oct 2011) $
 Revision          : $Revision: 7449 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef ITKMIDASIMAGEUPDATEPASTEREGIONPROCESSOR_H
#define ITKMIDASIMAGEUPDATEPASTEREGIONPROCESSOR_H

#include "itkMIDASImageUpdateRegionProcessor.h"

namespace itk
{

/**
 * \class MIDASImageUpdatePasteRegionProcessor
 * \brief Class to support undo/redo of a paste operation, within a given region,
 * where we take non-zero pixels in the source image, and write them to the destination image.
 */
template <class TPixel, unsigned int VImageDimension>
class ITK_EXPORT MIDASImageUpdatePasteRegionProcessor : public MIDASImageUpdateRegionProcessor<TPixel, VImageDimension> {

public:

  /** Standard class typedefs */
  typedef MIDASImageUpdatePasteRegionProcessor                     Self;
  typedef MIDASImageUpdateRegionProcessor<TPixel, VImageDimension> Superclass;
  typedef SmartPointer<Self>                                       Pointer;
  typedef SmartPointer<const Self>                                 ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(MIDASImageUpdatePasteRegionProcessor, MIDASImageUpdateRegionProcessor);

  /** Additional typedefs */
  typedef TPixel PixelType;
  typedef Image<TPixel, VImageDimension>  ImageType;
  typedef typename ImageType::Pointer     ImagePointer;
  typedef typename ImageType::IndexType   IndexType;
  typedef typename ImageType::SizeType    SizeType;
  typedef typename ImageType::RegionType  RegionType;

  /** Set the source image. Data is copied from here to destination image. */
  itkSetObjectMacro(SourceImage, ImageType);
  itkGetObjectMacro(SourceImage, ImageType);

  /** Set the source region of interest. */
  itkSetMacro(SourceRegionOfInterest, RegionType);
  itkGetMacro(SourceRegionOfInterest, RegionType);

  /** Overloaded method to provide simple acess via a std::vector, where we assume the length is 6 corresponding to the first 3 numbers indicating the starting index, and the next 3 numbers indicating the region size. */
  void SetSourceRegionOfInterest(std::vector<int> &region);

  /** Set/Get flag to copy background. If true, the background value of 0 is copied across, if false, only non-zero values are copied across. */
  itkSetMacro(CopyBackground, bool);
  itkGetMacro(CopyBackground, bool);

protected:
  MIDASImageUpdatePasteRegionProcessor();
  void PrintSelf(std::ostream& os, Indent indent) const;
  virtual ~MIDASImageUpdatePasteRegionProcessor() {}

  // This method that applies the change.
  virtual void ApplyUpdateToAfterImage();

private:
  MIDASImageUpdatePasteRegionProcessor(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  ImagePointer m_SourceImage;
  RegionType   m_SourceRegionOfInterest;

  bool m_CopyBackground;
};

} // end namespace

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMIDASImageUpdatePasteRegionProcessor.txx"
#endif

#endif // ITKMIDASIMAGEUPDATEPASTEREGIONPROCESSOR_H
