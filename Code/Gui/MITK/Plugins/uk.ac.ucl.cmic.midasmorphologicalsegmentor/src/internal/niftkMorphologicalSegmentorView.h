/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __niftkMorphologicalSegmentorView_h
#define __niftkMorphologicalSegmentorView_h

#include <niftkBaseSegmentorView.h>

#include <mitkImage.h>

#include <MorphologicalSegmentorPipelineParams.h>
#include "niftkMorphologicalSegmentorPreferencePage.h"

class niftkMorphologicalSegmentorController;

/**
 * \class niftkMorphologicalSegmentorView
 * \brief Provides the plugin component for the MIDAS brain segmentation functionality, originally developed at the Dementia Research Centre UCL.
 *
 * This plugin implements the paper:
 *
 * "Interactive algorithms for the segmentation and quantification of 3-D MRI brain scans"
 * by P. A. Freeborough, N. C. Fox and R. I. Kitney, published in
 * Computer Methods and Programs in Biomedicine 53 (1997) 15-25.
 *
 * \ingroup uk_ac_ucl_cmic_midasmorphologicalsegmentor_internal
 *
 * \sa niftkBaseSegmentorView
 * \sa niftkMorphologicalSegmentorPipelineManager
 * \sa MorphologicalSegmentorPipeline
 * \sa MorphologicalSegmentorPipelineInterface
 * \sa MorphologicalSegmentorPipelineParams
 */
class niftkMorphologicalSegmentorView : public niftkBaseSegmentorView
{
  Q_OBJECT

public:

  /// \brief Constructor, but most GUI construction is done in CreateQtPartControl().
  niftkMorphologicalSegmentorView();

  /// \brief Copy constructor which deliberately throws a runtime exception, as no-one should call it.
  niftkMorphologicalSegmentorView(const niftkMorphologicalSegmentorView& other);

  /// \brief Destructor.
  virtual ~niftkMorphologicalSegmentorView();

  /// \brief Each View for a plugin has its own globally unique ID.
  static const std::string VIEW_ID;

  /// \brief Returns VIEW_ID = uk.ac.ucl.cmic.midasmorphologicalsegmentor.
  virtual std::string GetViewID() const;

protected:

  /// \brief Creates the morphological segmentor controller that realises the GUI logic behind the view.
  virtual niftkBaseSegmentorController* CreateSegmentorController() override;

  /// \brief Called by framework, sets the focus on a specific widget, but currently does nothing.
  virtual void SetFocus() override;

  /// \brief Called when a node is removed.
  virtual void NodeRemoved(const mitk::DataNode* node) override;

  /// \brief Returns the name of the preferences node to look up.
  virtual QString GetPreferencesNodeName() override;

private:

  /// \brief The morphological segmentor controller that realises the GUI logic behind the view.
  niftkMorphologicalSegmentorController* m_MorphologicalSegmentorController;

};

#endif
