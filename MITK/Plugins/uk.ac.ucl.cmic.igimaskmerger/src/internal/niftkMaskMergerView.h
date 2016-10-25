/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkMaskMergerView_h
#define niftkMaskMergerView_h

#include <niftkBaseView.h>
#include <service/event/ctkEvent.h>

namespace niftk
{

class MaskMergerController;

/// \class MaskMergerView
/// \brief Provides a view to select mask images to combine together.
///
/// \sa niftkBaseView
/// \sa MaskMergerController
class MaskMergerView : public BaseView
{
  Q_OBJECT

public:

  /// \brief Each View for a plugin has its own globally unique ID, this one is
  /// "uk.ac.ucl.cmic.igimaskmerger" and the .cxx file and plugin.xml should match.
  static const QString VIEW_ID;

  /// \brief Constructor.
  MaskMergerView();

  /// \brief Copy constructor which deliberately throws a runtime exception, as no-one should call it.
  MaskMergerView(const MaskMergerView& other);

  /// \brief Destructor.
  virtual ~MaskMergerView();

protected:

  /// \brief Creates the GUI parts.
  virtual void CreateQtPartControl(QWidget* parent) override;

  /// \brief Called by framework, this method can set the focus on a specific widget,
  /// but we currently do nothing.
  virtual void SetFocus() override;

private slots:

  /// \brief We listen to the event bus to trigger updates.
  void OnUpdate(const ctkEvent& event);

private:

  /// \brief The Caffe segmentor controller that realises the GUI logic behind the view.
  QScopedPointer<MaskMergerController> m_MaskMergerController;
};

} // end namespace

#endif
