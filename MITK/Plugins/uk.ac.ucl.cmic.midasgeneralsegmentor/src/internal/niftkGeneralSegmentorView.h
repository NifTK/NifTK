/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkGeneralSegmentorView_h
#define niftkGeneralSegmentorView_h

#include <niftkBaseSegmentorView.h>

namespace niftk
{

class GeneralSegmentorController;

/// \class GeneralSegmentorView
/// \brief Provides a view for the MIDAS general purpose, Irregular Volume Editor functionality, originally developed
/// at the Dementia Research Centre UCL (http://dementia.ion.ucl.ac.uk/).
///
/// \sa niftkBaseSegmentorView
/// \sa GeneralSegmentorController
/// \sa niftkMorphologicalSegmentorView
class GeneralSegmentorView : public BaseSegmentorView
{
  Q_OBJECT

public:

  /// \brief Constructor.
  GeneralSegmentorView();

  /// \brief Copy constructor which deliberately throws a runtime exception, as no-one should call it.
  GeneralSegmentorView(const GeneralSegmentorView& other);

  /// \brief Destructor.
  virtual ~GeneralSegmentorView();

  /// \brief Each View for a plugin has its own globally unique ID, this one is
  /// "uk.ac.ucl.cmic.midasgeneralsegmentor" and the .cxx file and plugin.xml should match.
  static const QString VIEW_ID;

  /// \brief Returns the VIEW_ID = "uk.ac.ucl.cmic.midasgeneralsegmentor".
  virtual QString GetViewID() const;

protected:

  /// \brief Creates the general segmentor controller that realises the GUI logic behind the view.
  virtual BaseSegmentorController* CreateSegmentorController() override;

  /// \brief Called by framework, this method can set the focus on a specific widget,
  /// but we currently do nothing.
  virtual void SetFocus() override;

  /// \brief Returns the name of the preferences node to look up.
  /// \see niftkBaseSegmentorView::GetPreferencesNodeName
  virtual QString GetPreferencesNodeName() override;

private:

  /// \brief The general segmentor controller that realises the GUI logic behind the view.
  GeneralSegmentorController* m_GeneralSegmentorController;

};

}

#endif
