/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkDistanceMeasurerView_h
#define niftkDistanceMeasurerView_h

#include <niftkBaseView.h>
#include <service/event/ctkEvent.h>

namespace niftk
{

class DistanceMeasurerController;

/// \class DistanceMeasurerView
/// \brief Provides a view to measure distance to surfaces.
///
/// \sa niftkBaseView
/// \sa DistanceMeasurerController
class DistanceMeasurerView : public BaseView
{
  Q_OBJECT

public:

  /// \brief Each View for a plugin has its own globally unique ID, this one is
  /// "uk.ac.ucl.cmic.igidistancemeasurer" and the .cxx file and plugin.xml should match.
  static const QString VIEW_ID;

  /// \brief Constructor.
  DistanceMeasurerView();

  /// \brief Copy constructor which deliberately throws a runtime exception, as no-one should call it.
  DistanceMeasurerView(const DistanceMeasurerView& other);

  /// \brief Destructor.
  virtual ~DistanceMeasurerView();

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
  QScopedPointer<DistanceMeasurerController> m_DistanceMeasurerController;
};

} // end namespace

#endif
