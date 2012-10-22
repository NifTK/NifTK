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

#ifndef ITKMIDASIMAGEUPDATEPIXELWISESINGLEVALUEPROCESSOR_H
#define ITKMIDASIMAGEUPDATEPIXELWISESINGLEVALUEPROCESSOR_H

#include "itkMIDASImageUpdateProcessor.h"

namespace itk
{

/**
 * \class MIDASImageUpdatePixelWiseSingleValueProcessor
 * \brief Class to support undo/redo of an operation that
 * takes a list of pixels, and sets them all to a given value.
 *
 * This operation is used in the MIDAS PaintbrushTool, used
 * in the MorphologicalEditor.
 */
template <class TPixel, unsigned int VImageDimension>
class ITK_EXPORT MIDASImageUpdatePixelWiseSingleValueProcessor : public MIDASImageUpdateProcessor<TPixel, VImageDimension> {

public:

  /** Standard class typedefs */
  typedef MIDASImageUpdatePixelWiseSingleValueProcessor      Self;
  typedef MIDASImageUpdateProcessor<TPixel, VImageDimension> Superclass;
  typedef SmartPointer<Self>                                 Pointer;
  typedef SmartPointer<const Self>                           ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(MIDASImageUpdatePixelWiseSingleValueProcessor, MIDASImageUpdateProcessor);

  /** Additional typedefs */
  typedef TPixel PixelType;
  typedef Image<TPixel, VImageDimension>  ImageType;
  typedef typename ImageType::Pointer     ImagePointer;
  typedef typename ImageType::IndexType   IndexType;
  typedef typename ImageType::SizeType    SizeType;
  typedef typename ImageType::RegionType  RegionType;
  typedef std::vector<IndexType>          IndexListType;
  typedef std::vector<TPixel>             DataListType;

  /** Set/Get the pixel value to update. */
  itkSetMacro(Value, PixelType);
  itkGetMacro(Value, PixelType);

  /** Clears the list of indexes. */
  void ClearList();

  /** Adds a voxel to the end of the list. */
  void AddToList(IndexType &voxelIndex);

  /** Returns the number of voxels currently stored. */
  unsigned long int GetNumberOfVoxels();

  /** Returns the minimal bounding box of the contained voxels. */
  std::vector<int> ComputeMinimalBoundingBox();

  /** \see ImageUpdateProcessor::Undo() */
  virtual void Undo();

  /** \see ImageUpdateProcessor::Redo() */
  virtual void Redo();

protected:
  MIDASImageUpdatePixelWiseSingleValueProcessor();
  void PrintSelf(std::ostream& os, Indent indent) const;
  virtual ~MIDASImageUpdatePixelWiseSingleValueProcessor() {}

private:
  MIDASImageUpdatePixelWiseSingleValueProcessor(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  void ApplyListToDestinationImage(const DataListType& list);

  PixelType m_Value;
  bool m_UpdateCalculated;
  IndexListType m_Indexes;
  DataListType m_Before;
  DataListType m_After;
};

} // end namespace

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMIDASImageUpdatePixelWiseSingleValueProcessor.txx"
#endif

#endif // ITKMIDASIMAGEUPDATEPIXELWISESINGLEVALUEPROCESSOR_H
