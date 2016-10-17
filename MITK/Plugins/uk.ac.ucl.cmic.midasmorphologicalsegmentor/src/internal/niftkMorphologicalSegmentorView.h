/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkMorphologicalSegmentorView_h
#define niftkMorphologicalSegmentorView_h

#include <niftkBaseSegmentorView.h>

namespace niftk
{

/// \class MorphologicalSegmentorView
/// \brief Provides the plugin component for the MIDAS brain segmentation functionality, originally developed at the Dementia Research Centre UCL.
///
/// This plugin implements the paper:
///
/// "Interactive algorithms for the segmentation and quantification of 3-D MRI brain scans"
/// by P. A. Freeborough, N. C. Fox and R. I. Kitney, published in
/// Computer Methods and Programs in Biomedicine 53 (1997) 15-25.
///
/// \ingroup uk_ac_ucl_cmic_midasmorphologicalsegmentor_internal
///
/// \sa BaseSegmentorView
/// \sa niftkMorphologicalSegmentorPipelineManager
/// \sa MorphologicalSegmentorPipeline
/// \sa MorphologicalSegmentorPipelineInterface
/// \sa MorphologicalSegmentorPipelineParams
class MorphologicalSegmentorView : public BaseSegmentorView
{
  Q_OBJECT

public:

  /// \brief Constructor, but most GUI construction is done in CreateQtPartControl().
  MorphologicalSegmentorView();

  /// \brief Copy constructor which deliberately throws a runtime exception, as no-one should call it.
  MorphologicalSegmentorView(const MorphologicalSegmentorView& other);

  /// \brief Destructor.
  virtual ~MorphologicalSegmentorView();

  /// \brief Each View for a plugin has its own globally unique ID.
  static const QString VIEW_ID;

  /// \brief Returns VIEW_ID = uk.ac.ucl.cmic.midasmorphologicalsegmentor.
  virtual QString GetViewID() const;

protected:

  /// \brief Creates the morphological segmentor controller that realises the GUI logic behind the view.
  virtual BaseSegmentorController* CreateSegmentorController() override;

  /// \brief Called by framework, sets the focus on a specific widget, but currently does nothing.
  virtual void SetFocus() override;

  /// \brief Returns the name of the preferences node to look up.
  virtual QString GetPreferencesNodeName() override;

};

}

#endif
