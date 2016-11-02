/*=============================================================================

 KMaps:     An image processing toolkit for DCE-MRI analysis developed
            at the Molecular Imaging Center at University of Torino.

 See:       http://www.cim.unito.it

 Author:    Miklos Espak <espakm@gmail.com>

 Copyright (c) Miklos Espak
 All Rights Reserved.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef IntensityProfileView_h
#define IntensityProfileView_h

#include <berryISelectionListener.h>

#include "ui_IntensityProfileView.h"

#include <mitkImageStatisticsCalculator.h>
#include <mitkNodePredicateDataType.h>
#include <mitkNodePredicateProperty.h>
#include <mitkNodePredicateAnd.h>
#include <mitkNodePredicateNot.h>
#include <mitkNodePredicateDimension.h>
#include <mitkNodePredicateData.h>
#include <mitkPointSet.h>
#include <mitkIRenderWindowPartListener.h>

#include <itkFixedArray.h>

#include <niftkBaseView.h>

#include "internal/niftkVisibilityChangeObserver.h"

namespace mitk {
class DataNode;
}

namespace niftk
{

class IntensityProfileViewPrivate;

/*!
 \brief IntensityProfileView

 \warning  This application module is not yet documented.
 */
class IntensityProfileView :
    public niftk::BaseView,
    public VisibilityChangeObserver,
    public mitk::IRenderWindowPartListener
{
  // this is needed for all Qt objects that should have a Qt meta-object
  // (everything that derives from QObject and wants to have signal/slots)
    Q_OBJECT

public:

  typedef itk::FixedArray<mitk::ScalarType, 2> RangeBounds;

  static const std::string VIEW_ID;

  IntensityProfileView();
  virtual ~IntensityProfileView();

  virtual void CreateQtPartControl(QWidget *parent) override;

  /// \brief Called by framework, sets the focus on a specific widget.
  virtual void SetFocus() override;


  void plotProfileNode(mitk::DataNode::Pointer selectedNode);

  ///
  /// Called when the visibility of a node in the data storage changed
  ///
  virtual void OnVisibilityChanged(const mitk::DataNode* node) override;

  virtual void NodeAdded(const mitk::DataNode* node) override;
  virtual void NodeRemoved(const mitk::DataNode* node) override;
  virtual void NodeChanged(const mitk::DataNode* node) override;

  virtual void RenderWindowPartActivated(mitk::IRenderWindowPart* renderWindowPart) override;
  virtual void RenderWindowPartDeactivated(mitk::IRenderWindowPart* renderWindowPart) override;

protected:
  bool eventFilter(QObject *obj, QEvent *event) override;

private slots:
  void onProfileNodeChanged();
  void on_storeCrosshairButton_clicked();
  void on_storeStatisticsButton_clicked();
  void on_copyStatisticsButton_clicked();
  void on_clearCacheButton_clicked();
  void onCrosshairPositionEventDelayed();
  void calculateCrosshairProfiles(mitk::Point3D crosshairPos);
  void calculateRoiStatistics(mitk::DataNode* node, mitk::DataNode* roiNode);
  void plotRoiProfiles(mitk::DataNode* node);
  void plotRoiProfile(mitk::DataNode* node, mitk::DataNode* roi);
  void plotStoredProfiles();

private:

  void onNodeAddedInternal(const mitk::DataNode* node);
  void onNodeRemovedInternal(const mitk::DataNode* node);

  void onCrosshairVisibilityOn();
  void onCrosshairVisibilityOff();

  void onCrosshairPositionEvent();
  void onVisibilityOn(const mitk::DataNode* node);
  void onVisibilityOff(const mitk::DataNode* node);
  void selectNode(mitk::DataNode* node);
  void deselectNode(mitk::DataNode* node);
  mitk::TimeBounds ComputeTimeBounds();
  RangeBounds ComputeRangeBounds();
  void initPlotter();
  bool dimensionsAreEqual(const mitk::DataNode* node1, const mitk::DataNode* node2, bool discardTimeSteps);
  bool askXValues(mitk::DataNode* node);
  void getXValues(mitk::DataNode* node, QVector<double>& xValues, QVector<unsigned>& xValueOrder);

  void setDefaultLevelWindow(mitk::DataNode* node);

  typedef mitk::ImageStatisticsCalculator::Statistics Statistics;

  static mitk::NodePredicateDimension::Pointer is4DImage;
  static mitk::NodePredicateProperty::Pointer isIntensityProfile;
  static mitk::NodePredicateProperty::Pointer isVisible;
  static mitk::NodePredicateAnd::Pointer isVisibleIntensityProfile;
  static mitk::TNodePredicateDataType<mitk::Image>::Pointer hasImage;
  static mitk::NodePredicateProperty::Pointer isBinary;
  static mitk::NodePredicateNot::Pointer isNotBinary;
  static mitk::NodePredicateAnd::Pointer hasBinaryImage;
  static mitk::NodePredicateAnd::Pointer has4DBinaryImage;
  static mitk::TNodePredicateDataType<mitk::PointSet>::Pointer hasPointSet;
  static mitk::NodePredicateAnd::Pointer is4DNotBinaryImage;
  static mitk::NodePredicateAnd::Pointer isCrosshair;
  static mitk::NodePredicateData::Pointer isGroup;
  static mitk::NodePredicateAnd::Pointer isStudy;

  QScopedPointer<IntensityProfileViewPrivate> d_ptr;
  Ui::IntensityProfileView* ui;

  Q_DECLARE_PRIVATE(IntensityProfileView)
  Q_DISABLE_COPY(IntensityProfileView)

};

}

#endif
