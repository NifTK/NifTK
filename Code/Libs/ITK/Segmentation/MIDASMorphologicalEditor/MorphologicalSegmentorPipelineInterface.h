/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef _MIDASMORPHOLOGICALSEGMENTORPIPELINEINTERFACE_H_INCLUDED
#define _MIDASMORPHOLOGICALSEGMENTORPIPELINEINTERFACE_H_INCLUDED

#include "MorphologicalSegmentorPipelineParams.h"

/**
 * \class MorphologicalSegmentorPipelineInterface
 * \brief Abstract interface to plug ITK pipeline into MITK framework to represent the MIDAS Morphological Segmentor Pipeline.
 *
 * \ingroup midas_morph_editor
 */
class MorphologicalSegmentorPipelineInterface
{
public:

  /// \brief Default no-op constructor.
  MorphologicalSegmentorPipelineInterface() {};

  /// \brief Default no-op destructor.
  ~MorphologicalSegmentorPipelineInterface() {};

  /// \brief Update the pipeline.
  ///
  /// \param editingFlags array of 4 booleans to say which images are being editted.
  /// \param editingRegion pass in an array of 6 integers containing size[0-2], and index[3-5] for the region being edited.
  virtual void Update(std::vector<bool>& editingFlags, std::vector<int>& editingRegion) = 0;
};

#endif
