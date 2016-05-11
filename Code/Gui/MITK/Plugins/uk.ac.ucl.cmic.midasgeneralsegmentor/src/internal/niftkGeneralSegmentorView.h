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

class niftkGeneralSegmentorController;

/**
 * \class niftkGeneralSegmentorView
 * \brief Provides a view for the MIDAS general purpose, Irregular Volume Editor functionality, originally developed
 * at the Dementia Research Centre UCL (http://dementia.ion.ucl.ac.uk/).
 *
 * \sa niftkBaseSegmentorView
 * \sa niftkGeneralSegmentorController
 * \sa niftkMorphologicalSegmentorView
 */
class niftkGeneralSegmentorView : public niftkBaseSegmentorView
{
  Q_OBJECT

public:

  /// \brief Constructor.
  niftkGeneralSegmentorView();

  /// \brief Copy constructor which deliberately throws a runtime exception, as no-one should call it.
  niftkGeneralSegmentorView(const niftkGeneralSegmentorView& other);

  /// \brief Destructor.
  virtual ~niftkGeneralSegmentorView();

  /// \brief Each View for a plugin has its own globally unique ID, this one is
  /// "uk.ac.ucl.cmic.midasgeneralsegmentor" and the .cxx file and plugin.xml should match.
  static const std::string VIEW_ID;

  /// \brief Returns the VIEW_ID = "uk.ac.ucl.cmic.midasgeneralsegmentor".
  virtual std::string GetViewID() const;

protected:

  /// \see mitk::ILifecycleAwarePart::PartVisible
  virtual void Visible() override;

  /// \see mitk::ILifecycleAwarePart::PartHidden
  virtual void Hidden() override;

  /// \brief Creates the general segmentor controller that realises the GUI logic behind the view.
  virtual niftkBaseSegmentorController* CreateSegmentorController() override;

  /// \brief Called by framework, this method can set the focus on a specific widget,
  /// but we currently do nothing.
  virtual void SetFocus() override;

  /// \brief Returns the name of the preferences node to look up.
  /// \see niftkBaseSegmentorView::GetPreferencesNodeName
  virtual QString GetPreferencesNodeName() override;

  /// \brief This view registers with the mitk::DataStorage and listens for changing
  /// data, so this method is called when any node is changed, but only performs an update,
  /// if the nodes changed are those registered with the ToolManager as WorkingData,
  /// see class introduction.
  virtual void NodeChanged(const mitk::DataNode* node) override;

  /// \brief This view registers with the mitk::DataStorage and listens for removing
  /// data, so this method cancels the operation and frees the resources if the
  /// segmentation node is removed.
  virtual void NodeRemoved(const mitk::DataNode* node) override;

  void onVisibilityChanged(const mitk::DataNode* node) override;

private:

  /// \brief The general segmentor controller that realises the GUI logic behind the view.
  niftkGeneralSegmentorController* m_GeneralSegmentorController;

};

#endif
