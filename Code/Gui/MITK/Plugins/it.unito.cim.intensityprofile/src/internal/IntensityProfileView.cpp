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

#include "IntensityProfileView.h"

#include <itkSimpleDataObjectDecorator.h>

#include <mitkItkBaseDataAdapter.h>
#include <mitkImageTimeSelector.h>
#include <mitkColorProperty.h>
#include <mitkImageStatisticsHolder.h>
#include <mitkProperties.h>
#include <mitkLookupTable.h>

#include <QmitkStdMultiWidget.h>
#include <QmitkNodeDescriptorManager.h>

#include <qwt_plot_curve.h>
#include <qwt_legend.h>
#include <qwt_symbol.h>
#include <qwt_scale_widget.h>

#include <QObject>
#include <QVector>
#include <QSet>
#include <QMessageBox>
#include <QClipboard>
#include <QToolTip>
#include <QEvent>
#include <QWheelEvent>
#include <QInputDialog>

void MedianHybridQuickSort(std::vector<double> array, std::vector<unsigned>& array2);

class IntensityProfileViewPrivate {

  class StaticInit {
  public:
    StaticInit() {
      IntensityProfileViewPrivate::staticInit();
    }
  };
  static StaticInit staticInit2;
  static void staticInit();

public:
  IntensityProfileViewPrivate();

  static const int MaxSymbols = 8;
  static QwtSymbol Symbols[MaxSymbols];

  QWidget* m_Parent;

//  static QwtSymbol::Style SymbolStyles[16];

  bool showCrosshairProfile;
  bool crosshairPositionListenerIsAdded;
  bool showRoiProfiles;

  mitk::MessageDelegate<IntensityProfileView>* crosshairPositionListener;
  bool pendingCrosshairPositionEvent;

  QList<mitk::DataNode::Pointer> referenceNodes;
  QMap<mitk::DataNode*, unsigned long> levelWindowObserverTags;

  mitk::IRenderWindowPart* display;

  itk::SimpleMemberCommand<IntensityProfileView>::Pointer levelWindowModifiedCommand;
  itk::SimpleMemberCommand<IntensityProfileView>::Pointer profileNodeChangedCommand;

  IntensityProfileView::RangeBounds rangeBounds;

  mitk::Index3D crosshairIndex;
  QMap<mitk::DataNode*, QwtPlotCurve*> crosshairProfiles;
  QMap<QString, QwtPlotCurve*> keptCrosshairProfiles;

//  QMap<mitk::DataNode*, std::vector<std::vector<IntensityProfileView::Statistics> > > statistics;
  typedef IntensityProfileView::Statistics Statistics;
  typedef std::vector<Statistics> StatisticsAtTimeSteps;
  typedef QMap<mitk::DataNode*, QMap<mitk::DataNode*, StatisticsAtTimeSteps> > RoiStatisticsMap;
  RoiStatisticsMap statisticsByNodeAndRoi;
  RoiStatisticsMap statisticsByRoiAndNode;

  QList<mitk::DataNode::Pointer> roiNodes;
  // Profiles per reference node and per ROI
  typedef QMap<mitk::DataNode*, QwtPlotCurve*> RoiProfileMap;
  typedef QMap<mitk::DataNode*, RoiProfileMap> RoiProfileMapsByNode;
  RoiProfileMapsByNode roiProfileMapsByNode;

  QList<mitk::DataNode::Pointer> profileNodes;
  QMap<mitk::DataNode*, unsigned long> profileNodeChangeObservers;
  QList<QwtPlotCurve*> storedProfileCurves;

  QwtScaleWidget* xAxis;
  QwtScaleWidget* yAxis;

};

QwtSymbol IntensityProfileViewPrivate::Symbols[MaxSymbols];
IntensityProfileViewPrivate::StaticInit IntensityProfileViewPrivate::staticInit2;

void
IntensityProfileViewPrivate::staticInit()
{
//  Symbols[0].setPen(QColor(Qt::red));
//  Symbols[1].setPen(QColor(Qt::blue));
//  Symbols[2].setPen(QColor(Qt::magenta));
//  Symbols[3].setPen(QColor(Qt::green));
//  Symbols[4].setPen(QColor(Qt::darkRed));
//  Symbols[5].setPen(QColor(Qt::darkBlue));
//  Symbols[6].setPen(QColor(Qt::darkMagenta));
//  Symbols[7].setPen(QColor(Qt::darkGreen));
//  Symbols[0].setBrush(QColor(Qt::red));
//  Symbols[1].setBrush(QColor(Qt::blue));
//  Symbols[2].setBrush(QColor(Qt::magenta));
//  Symbols[3].setBrush(QColor(Qt::green));
//  Symbols[4].setBrush(QColor(Qt::darkRed));
//  Symbols[5].setBrush(QColor(Qt::darkBlue));
//  Symbols[6].setBrush(QColor(Qt::darkMagenta));
//  Symbols[7].setBrush(QColor(Qt::darkGreen));
  Symbols[0].setStyle(QwtSymbol::Ellipse);
  Symbols[1].setStyle(QwtSymbol::Rect);
  Symbols[2].setStyle(QwtSymbol::Diamond);
  Symbols[3].setStyle(QwtSymbol::Triangle);
  Symbols[4].setStyle(QwtSymbol::Ellipse);
  Symbols[5].setStyle(QwtSymbol::Rect);
  Symbols[6].setStyle(QwtSymbol::Diamond);
  Symbols[7].setStyle(QwtSymbol::Triangle);
  Symbols[0].setSize(5);
  Symbols[1].setSize(5);
  Symbols[2].setSize(5);
  Symbols[3].setSize(5);
  Symbols[4].setSize(5);
  Symbols[5].setSize(5);
  Symbols[6].setSize(5);
  Symbols[7].setSize(5);
}

IntensityProfileViewPrivate::IntensityProfileViewPrivate()
: pendingCrosshairPositionEvent(false)
{
}

const std::string IntensityProfileView::VIEW_ID =
    "it.unito.cim.IntensityProfileView";

mitk::NodePredicateDimension::Pointer IntensityProfileView::is4DImage =
    mitk::NodePredicateDimension::New(4);

mitk::NodePredicateProperty::Pointer IntensityProfileView::isIntensityProfile =
    mitk::NodePredicateProperty::New("intensity profile");

mitk::NodePredicateProperty::Pointer IntensityProfileView::isVisible =
    mitk::NodePredicateProperty::New("visible", mitk::BoolProperty::New(true));

mitk::NodePredicateAnd::Pointer IntensityProfileView::isVisibleIntensityProfile =
    mitk::NodePredicateAnd::New(isVisible, isIntensityProfile);

mitk::TNodePredicateDataType<mitk::Image>::Pointer IntensityProfileView::hasImage =
    mitk::TNodePredicateDataType<mitk::Image>::New();

mitk::NodePredicateProperty::Pointer IntensityProfileView::isBinary =
    mitk::NodePredicateProperty::New("binary", mitk::BoolProperty::New(true));

mitk::NodePredicateNot::Pointer IntensityProfileView::isNotBinary =
    mitk::NodePredicateNot::New(isBinary);

mitk::NodePredicateAnd::Pointer IntensityProfileView::is4DNotBinaryImage =
    mitk::NodePredicateAnd::New(is4DImage, isNotBinary);

mitk::NodePredicateAnd::Pointer IntensityProfileView::hasBinaryImage =
    mitk::NodePredicateAnd::New(hasImage, isBinary);

mitk::NodePredicateAnd::Pointer IntensityProfileView::has4DBinaryImage =
    mitk::NodePredicateAnd::New(hasBinaryImage, is4DImage);

mitk::TNodePredicateDataType<mitk::PointSet>::Pointer IntensityProfileView::hasPointSet =
    mitk::TNodePredicateDataType<mitk::PointSet>::New();

mitk::NodePredicateAnd::Pointer IntensityProfileView::isCrosshair =
    mitk::NodePredicateAnd::New(
        mitk::NodePredicateProperty::New("name", mitk::StringProperty::New("widget1Plane")),
        mitk::NodePredicateProperty::New("helper object", mitk::BoolProperty::New(true)));

mitk::NodePredicateData::Pointer IntensityProfileView::isGroup =
    mitk::NodePredicateData::New(0);

mitk::NodePredicateAnd::Pointer IntensityProfileView::isStudy =
    mitk::NodePredicateAnd::New(
        isGroup,
        mitk::NodePredicateProperty::New("Study"));

IntensityProfileView::IntensityProfileView()
: d_ptr(new IntensityProfileViewPrivate()),
  ui(0)
{
  Q_D(IntensityProfileView);
  d->pendingCrosshairPositionEvent = false;
  d->showCrosshairProfile = true;
  d->crosshairPositionListenerIsAdded = false;
  d->showRoiProfiles = true;
  d->crosshairPositionListener = new mitk::MessageDelegate<IntensityProfileView>(this, &IntensityProfileView::onCrosshairPositionEvent);

  d->display = 0;

  d->levelWindowModifiedCommand = itk::SimpleMemberCommand<IntensityProfileView>::New();
  d->levelWindowModifiedCommand->SetCallbackFunction(this, &IntensityProfileView::initPlotter);

  d->profileNodeChangedCommand = itk::SimpleMemberCommand<IntensityProfileView>::New();
  d->profileNodeChangedCommand->SetCallbackFunction(this, &IntensityProfileView::onProfileNodeChanged);

  // Adding "Intensity Profiles"
  QmitkNodeDescriptorManager* nodeDescriptorManager = QmitkNodeDescriptorManager::GetInstance();
  mitk::NodePredicateProperty::Pointer isIntensityProfile = mitk::NodePredicateProperty::New("intensity profile");
  QmitkNodeDescriptor* profileNodeDescriptor =
      new QmitkNodeDescriptor(
          tr("IntensityProfile"),
          QString(":/it.unito.cim.intensityprofile/plot1.jpg"),
          isIntensityProfile,
          this);
  nodeDescriptorManager->AddDescriptor(profileNodeDescriptor);
}

IntensityProfileView::~IntensityProfileView()
{
  Q_D(IntensityProfileView);

  if (d->showCrosshairProfile) {
    onCrosshairVisibilityOff();
  }
  // TODO somewhere these observers should be removed. Maybe not here.
  foreach (mitk::DataNode* roiNode, d->roiNodes) {
    onVisibilityOff(roiNode);
  }
  foreach (mitk::DataNode* node, d->referenceNodes) {
    deselectNode(node);
  }

  delete ui;
}

void IntensityProfileView::CreateQtPartControl(QWidget *parent) {
  // build up qt view, unless already done
  if (ui) {
    return;
  }
  // create GUI widgets from the Qt Designer's .ui file
  ui = new Ui::IntensityProfileView();
  ui->setupUi(parent);

  ui->plotter->insertLegend(new QwtLegend(), QwtPlot::TopLegend);

  Q_D(IntensityProfileView);
  d->xAxis = ui->plotter->axisWidget(QwtPlot::xBottom);
  d->xAxis->installEventFilter(this);
  d->yAxis = ui->plotter->axisWidget(QwtPlot::yLeft);
  d->yAxis->installEventFilter(this);

  ui->storeStatisticsButton->hide();
  ui->storeCrosshairButton->hide();

//  connect(ui->storeCrosshairButton, SIGNAL(clicked()), SLOT(on_storeCrosshairButton_clicked()));
//  connect(ui->storeStatisticsButton, SIGNAL(clicked()), SLOT(on_storeStatisticsButton_clicked()));
  connect(ui->copyStatisticsButton, SIGNAL(clicked()), SLOT(on_copyStatisticsButton_clicked()));
  connect(ui->clearCacheButton, SIGNAL(clicked()), SLOT(on_clearCacheButton_clicked()));

  mitk::DataStorage* dataStorage = GetDataStorage();
  if (dataStorage) {
    mitk::DataStorage::SetOfObjects::ConstPointer visibleNodes =
        dataStorage->GetSubset(mitk::NodePredicateProperty::New("visible", mitk::BoolProperty::New(true)));
    mitk::DataStorage::SetOfObjects::ConstIterator it = visibleNodes->Begin();
    mitk::DataStorage::SetOfObjects::ConstIterator end = visibleNodes->End();
    while (it != end) {
      onVisibilityOn(it->Value());
      ++it;
    }
  }
  else {
    MITK_INFO << "IntensityProfileView() data storage not ready";
  }

  d->m_Parent = parent;
}

void
IntensityProfileView::SetFocus()
{
  ui->plotter->setFocus();
}

void
IntensityProfileView::onVisibilityChanged(const mitk::DataNode* node)
{
  if (node->IsVisible(0)) {
    onVisibilityOn(node);
  }
  else {
    onVisibilityOff(node);
  }
}

void
IntensityProfileView::onVisibilityOn(const mitk::DataNode* cnode)
{
  if (!cnode) {
    return;
  }

  Q_D(IntensityProfileView);

  mitk::DataNode* node = const_cast<mitk::DataNode*>(cnode);

  if (isCrosshair->CheckNode(node)) {
    onCrosshairVisibilityOn();
  }
  else if (is4DNotBinaryImage->CheckNode(node)) {
    if (d->referenceNodes.contains(node)) {
      return;
    }
    d->referenceNodes.push_back(node);
    selectNode(node);
  }
  else if (hasBinaryImage->CheckNode(node)) {
    if (d->roiNodes.contains(node)) {
      return;
    }
    d->roiNodes.push_back(node);
    foreach (mitk::DataNode* referenceNode, d->referenceNodes) {
      if (dimensionsAreEqual(referenceNode, node, true)) {
        plotRoiProfile(referenceNode, node);
      }
    }
  }
  else {
    return;
  }

  if (d->showCrosshairProfile && !d->crosshairPositionListenerIsAdded) {
    onCrosshairVisibilityOn();
  }
  initPlotter();
}

void
IntensityProfileView::onVisibilityOff(const mitk::DataNode* cnode)
{
  if (!cnode) {
    return;
  }

  Q_D(IntensityProfileView);

  mitk::DataNode* node = const_cast<mitk::DataNode*>(cnode);

  if (isCrosshair->CheckNode(node)) {
    onCrosshairVisibilityOff();
  }
  else if (is4DNotBinaryImage->CheckNode(node)) {
    if (!d->referenceNodes.contains(node)) {
      return;
    }
    deselectNode(node);
    d->referenceNodes.removeOne(node);
  }
  else if (hasBinaryImage->CheckNode(node)) {
    if (!d->roiNodes.contains(node)) {
      return;
    }
    d->roiNodes.removeOne(node);

    IntensityProfileViewPrivate::RoiProfileMapsByNode::iterator itRoiProfileMaps = d->roiProfileMapsByNode.begin();
    IntensityProfileViewPrivate::RoiProfileMapsByNode::iterator endRoiProfileMaps = d->roiProfileMapsByNode.end();
    while (itRoiProfileMaps != endRoiProfileMaps) {
      IntensityProfileViewPrivate::RoiProfileMap::iterator itRoiProfile = itRoiProfileMaps->find(node);
      itRoiProfileMaps->erase(itRoiProfile);
      delete *itRoiProfile;
      ++itRoiProfileMaps;
    }
  }

  initPlotter();
}

void
IntensityProfileView::NodeAdded(const mitk::DataNode* node)
{
  if (node->IsVisible(0)) {
    onVisibilityOn(node);
  }
}

void
IntensityProfileView::NodeRemoved(const mitk::DataNode* node)
{
  if (node->IsVisible(0)) {
    onVisibilityOff(node);
  }
}

void
IntensityProfileView::NodeChanged(const mitk::DataNode* node)
{
  Q_D(IntensityProfileView);

  if (is4DImage->CheckNode(node)) {
    mitk::DataNode* referenceNode = const_cast<mitk::DataNode*>(node);
    if (d->referenceNodes.contains(referenceNode)) {
      foreach (mitk::DataNode* roiNode, d->roiNodes) {
        delete d->roiProfileMapsByNode[referenceNode][roiNode];
      }
      d->roiProfileMapsByNode[referenceNode].clear();
      plotRoiProfiles(referenceNode);
    }
  }
  else if (hasBinaryImage->CheckNode(node)) {
    mitk::DataNode* roiNode = const_cast<mitk::DataNode*>(node);
    if (d->roiNodes.contains(roiNode)) {
      foreach (mitk::DataNode* referenceNode, d->referenceNodes) {
        if (dimensionsAreEqual(referenceNode, roiNode, true)) {
          IntensityProfileViewPrivate::RoiProfileMap roiProfiles = d->roiProfileMapsByNode[referenceNode];
          if (roiProfiles.contains(roiNode)) {
            delete roiProfiles[roiNode];
          }
          roiProfiles.remove(roiNode);
          plotRoiProfile(referenceNode, roiNode);
        }
      }
    }
  }
}

void
IntensityProfileView::initPlotter()
{
//  mitk::TimeBounds timeBounds = ComputeTimeBounds();
  RangeBounds rangeBounds = ComputeRangeBounds();
  ui->plotter->setAxisScale(QwtPlot::yLeft, rangeBounds[0], rangeBounds[1]);
  ui->plotter->replot();
}

mitk::TimeBounds
IntensityProfileView::ComputeTimeBounds()
{
  using namespace mitk;
  Q_D(IntensityProfileView);

  TimeBounds timeBounds;

  ScalarType stmin, stmax, cur;

  stmin= ScalarTypeNumericTraits::NonpositiveMin();
  stmax= ScalarTypeNumericTraits::max();

  timeBounds[0]=stmax; timeBounds[1]=stmin;


  foreach (mitk::DataNode* node, d->referenceNodes)
  {
    const Geometry3D* geometry = node->GetData()->GetUpdatedTimeSlicedGeometry();
    if (geometry != NULL )
    {
      const TimeBounds & curTimeBounds = geometry->GetTimeBounds();
      cur=curTimeBounds[0];
      //is it after -infinity, but before everything else that we found until now?
      if((cur > stmin) && (cur < timeBounds[0]))
        timeBounds[0] = cur;

      cur=curTimeBounds[1];
      //is it before infinity, but after everything else that we found until now?
      if((cur < stmax) && (cur > timeBounds[1]))
        timeBounds[1] = cur;
    }
  }
  if(!(timeBounds[0] < stmax))
  {
    timeBounds[0] = stmin;
    timeBounds[1] = stmax;
  }
  return timeBounds;
}

IntensityProfileView::RangeBounds
IntensityProfileView::ComputeRangeBounds()
{
  Q_D(IntensityProfileView);

  RangeBounds rangeBounds;

  mitk::ScalarType minRangeMin = mitk::ScalarTypeNumericTraits::max();
  mitk::ScalarType maxRangeMax = mitk::ScalarTypeNumericTraits::min();

  foreach (mitk::DataNode* node, d->referenceNodes) {
    mitk::LevelWindow levelWindow;

    if (node->GetLevelWindow(levelWindow)) {
      double rangeMin = levelWindow.GetRangeMin();
      if (rangeMin < minRangeMin) {
        minRangeMin = rangeMin;
      }
      double rangeMax = levelWindow.GetRangeMax();
      if (rangeMax > maxRangeMax) {
        maxRangeMax = rangeMax;
      }
    }
  }

  // If we have not found any level window:
  if (minRangeMin == mitk::ScalarTypeNumericTraits::max()) {
    rangeBounds[0] = 0.0;
    rangeBounds[1] = 100.0;
  }
  else {
    rangeBounds[0] = minRangeMin;
    rangeBounds[1] = maxRangeMax;
  }

  return rangeBounds;
}

bool
IntensityProfileView::dimensionsAreEqual(const mitk::DataNode* node1, const mitk::DataNode* node2, bool discardTimeSteps)
{
  mitk::Image* image1 = dynamic_cast<mitk::Image*>(node1->GetData());
  int dim1 = image1->GetDimension();
  mitk::Image* image2 = dynamic_cast<mitk::Image*>(node2->GetData());
  int dim2 = image2->GetDimension();
  if ((dim1 != 3 && dim1 != 4) || (dim2 != 3 && dim2 != 4)) {
    return false;
  }
  if (!discardTimeSteps &&
      (dim1 != dim2 || (dim1 == 4 && dim2 == 4 && dim1 != dim2))) {
      return false;
  }
  for (int i = 0; i < 3; ++i) {
    if (image1->GetDimension(i) != image2->GetDimension(i)) {
      return false;
    }
  }
  return true;
}

void
IntensityProfileView::selectNode(mitk::DataNode* node)
{
  Q_D(IntensityProfileView);

//  mitk::DataNode* selectedNode = node;
//  mitk::DataNode* referenceNode = selectedNode;

  // The dimension must equal for every selected 4D image,
  // otherwise skipped.
  if (!d->referenceNodes.isEmpty()) {
    if (!dimensionsAreEqual(node, d->referenceNodes[0], false)) {
      return;
    }
  }

//  if (isIntensityProfile->CheckNode(selectedNode) ||
//      hasPointSet->CheckNode(selectedNode) ||
//      has4DBinaryImage->CheckNode(selectedNode)) {
//    mitk::DataStorage::SetOfObjects::ConstPointer selectedNodeSources =
//        GetDataStorage()->GetSources(selectedNode);
//    if (selectedNodeSources->Size() != 1) {
//      deselectNode();
//      d->referenceNode = 0;
//      d->selectedNode = 0;
//      return;
//    }
//    referenceNode = *selectedNodeSources->begin();
//  }
//  if (!is4DImage->CheckNode(referenceNode)) {
//    deselectNode();
//    d->referenceNode = 0;
//    d->selectedNode = 0;
//    return;
//  }
//
//  if (d->referenceNode == referenceNode) {
//    d->selectedNode = selectedNode;
//    return;
//  }
//  deselectNode();

  setDefaultLevelWindow(node);
  mitk::BaseProperty* levelWindowProperty = node->GetProperty("levelwindow");
  d->levelWindowObserverTags[node] = levelWindowProperty->AddObserver(itk::ModifiedEvent(), d->levelWindowModifiedCommand);

  if (d->showCrosshairProfile) {
    onCrosshairPositionEvent();
  }

  if (d->showRoiProfiles) {
    plotRoiProfiles(node);
  }

  mitk::DataStorage::SetOfObjects::ConstPointer profileNodeSet =
      GetDataStorage()->GetDerivations(node, isVisibleIntensityProfile);

  d->profileNodes.clear();
  mitk::DataStorage::SetOfObjects::const_iterator it = profileNodeSet->begin();
  mitk::DataStorage::SetOfObjects::const_iterator end = profileNodeSet->end();
  while (it != end) {
    mitk::DataNode* profileNode = *it;
    d->profileNodes.push_back(profileNode);
    ++it;
  }

  plotStoredProfiles();
  for (int i = 0; i < d->profileNodes.size(); ++i) {
    unsigned long observerTag = d->profileNodes[i]->AddObserver(
        itk::ModifiedEvent(),
        d->profileNodeChangedCommand);
    d->profileNodeChangeObservers[d->profileNodes[i]] = observerTag;
  }
}

void
IntensityProfileView::deselectNode(mitk::DataNode* node)
{
  Q_D(IntensityProfileView);

  mitk::BaseProperty* levelWindowProperty = node->GetProperty("levelwindow");
  levelWindowProperty->RemoveObserver(d->levelWindowObserverTags[node]);

  if (d->crosshairProfiles.contains(node)) {
    delete d->crosshairProfiles[node];
    d->crosshairProfiles.remove(node);
  }

  if (d->roiProfileMapsByNode.contains(node)) {
    foreach (QwtPlotCurve* profile, d->roiProfileMapsByNode[node]) {
      delete profile;
    }
    d->roiProfileMapsByNode[node].clear();
    d->roiProfileMapsByNode.remove(node);
  }

  foreach (mitk::DataNode* profileNode, d->profileNodeChangeObservers.keys()) {
    profileNode->RemoveObserver(d->profileNodeChangeObservers[profileNode]);
  }
  if (!d->storedProfileCurves.isEmpty()) {
    foreach (QwtPlotCurve* curve, d->storedProfileCurves) {
      delete curve;
    }
  }
  d->storedProfileCurves.clear();

  ui->plotter->replot();
}

void
IntensityProfileView::onCrosshairVisibilityOn()
{
  Q_D(IntensityProfileView);
  d->showCrosshairProfile = true;

//  QmitkStdMultiWidget* display = GetActiveStdMultiWidget(false);
  mitk::IRenderWindowPart* display = GetRenderWindowPart();
  if (display != d->display) {
    if (d->display && d->showCrosshairProfile) {
      d->display->GetRenderWindow("transversal")->GetSliceNavigationController()->crosshairPositionEvent.RemoveListener(*d->crosshairPositionListener);
      d->display->GetRenderWindow("sagittal")->GetSliceNavigationController()->crosshairPositionEvent.RemoveListener(*d->crosshairPositionListener);
      d->display->GetRenderWindow("coronal")->GetSliceNavigationController()->crosshairPositionEvent.RemoveListener(*d->crosshairPositionListener);
    }
    d->display = display;
  }
  if (!display) {
    return;
  }

  onCrosshairPositionEvent();
  display->GetRenderWindow("transversal")->GetSliceNavigationController()->crosshairPositionEvent.AddListener(*d->crosshairPositionListener);
  display->GetRenderWindow("sagittal")->GetSliceNavigationController()->crosshairPositionEvent.AddListener(*d->crosshairPositionListener);
  display->GetRenderWindow("coronal")->GetSliceNavigationController()->crosshairPositionEvent.AddListener(*d->crosshairPositionListener);
  d->crosshairPositionListenerIsAdded = true;
}

void
IntensityProfileView::onCrosshairVisibilityOff()
{
  Q_D(IntensityProfileView);
  d->showCrosshairProfile = false;

  mitk::IRenderWindowPart* display = GetRenderWindowPart();
  if (display != d->display) {
    if (d->display && d->showCrosshairProfile) {
      d->display->GetRenderWindow("transversal")->GetSliceNavigationController()->crosshairPositionEvent.RemoveListener(*d->crosshairPositionListener);
      d->display->GetRenderWindow("sagittal")->GetSliceNavigationController()->crosshairPositionEvent.RemoveListener(*d->crosshairPositionListener);
      d->display->GetRenderWindow("coronal")->GetSliceNavigationController()->crosshairPositionEvent.RemoveListener(*d->crosshairPositionListener);
    }
    d->display = display;
  }
  if (!display) {
    return;
  }

  display->GetRenderWindow("transversal")->GetSliceNavigationController()->crosshairPositionEvent.RemoveListener(*d->crosshairPositionListener);
  display->GetRenderWindow("sagittal")->GetSliceNavigationController()->crosshairPositionEvent.RemoveListener(*d->crosshairPositionListener);
  display->GetRenderWindow("coronal")->GetSliceNavigationController()->crosshairPositionEvent.RemoveListener(*d->crosshairPositionListener);
  d->crosshairPositionListenerIsAdded = true;

  d->crosshairProfiles.clear();
  ui->plotter->replot();
}

void
IntensityProfileView::onCrosshairPositionEvent()
{
  Q_D(IntensityProfileView);
  if (!d->pendingCrosshairPositionEvent) {
    d->pendingCrosshairPositionEvent = true;
    QTimer::singleShot(0, this, SLOT(onCrosshairPositionEventDelayed()));
  }
}

void
IntensityProfileView::onCrosshairPositionEventDelayed()
{
  Q_D(IntensityProfileView);
  d->pendingCrosshairPositionEvent = false;
  mitk::IRenderWindowPart* display = GetRenderWindowPart();
  if (display != d->display) {
    if (d->display && d->showCrosshairProfile) {
      d->display->GetRenderWindow("transversal")->GetSliceNavigationController()->crosshairPositionEvent.RemoveListener(*d->crosshairPositionListener);
      d->display->GetRenderWindow("sagittal")->GetSliceNavigationController()->crosshairPositionEvent.RemoveListener(*d->crosshairPositionListener);
      d->display->GetRenderWindow("coronal")->GetSliceNavigationController()->crosshairPositionEvent.RemoveListener(*d->crosshairPositionListener);
    }
    d->display = display;
  }
  if (!display) {
    return;
  }
  const mitk::Point3D crossPosition = display->GetSelectedPosition();
  calculateCrosshairProfiles(crossPosition);
  foreach (mitk::DataNode* node, d->referenceNodes) {
    d->crosshairProfiles[node]->attach(ui->plotter);
  }
  ui->plotter->replot();
}

void
IntensityProfileView::calculateCrosshairProfiles(mitk::Point3D crosshairPos)
{
  Q_D(IntensityProfileView);

  int symbolIndex = 0;
  foreach (mitk::DataNode* node, d->referenceNodes) {
    mitk::Image::Pointer image4D = dynamic_cast<mitk::Image*>(node->GetData());
    int timeSteps = image4D->GetTimeSteps();

    // TODO select the widest interval!
    // maybe the geometry should be calculated
//    mitk::LevelWindow levelWindow;
//    node->GetLevelWindow(levelWindow);
//    ui->plotter->setAxisScale(QwtPlot::yLeft, levelWindow.GetRangeMin(), levelWindow.GetRangeMax());
    //    ui->plotter->setAxisScale(QwtPlot::xBottom, 0.0, vfaTimeSteps - 1.0, 5.0);

    mitk::Index3D p;
    image4D->GetGeometry()->WorldToIndex(crosshairPos, p);
    d->crosshairIndex = p;

    if (!d->crosshairProfiles[node]) {
      QString profileName = QString("%1 [crosshair]").arg(QString::fromStdString(node->GetName()));
      d->crosshairProfiles[node] = new QwtPlotCurve(profileName);
    }
    d->crosshairProfiles[node]->setSymbol(IntensityProfileViewPrivate::Symbols[symbolIndex]);

    QVector<double> xValues(timeSteps);
    QVector<unsigned> xValueOrder(timeSteps);
    getXValues(node, xValues, xValueOrder);
    QVector<double> xValuesOrdered(timeSteps);
    QVector<double> yValues(timeSteps);
    for (int t = 0; t < timeSteps; ++t) {
      xValuesOrdered[t] = xValues[xValueOrder[t]];
      yValues[t] = image4D->GetPixelValueByIndex(p, xValueOrder[t]);
    }
    d->crosshairProfiles[node]->setData(xValuesOrdered, yValues);

    symbolIndex = (symbolIndex + 1) % IntensityProfileViewPrivate::MaxSymbols;
  }
}

void
IntensityProfileView::getXValues(mitk::DataNode* node, QVector<double>& xValues, QVector<unsigned>& xValueOrder) {
  bool newXValues = false;
  mitk::FloatLookupTableProperty::Pointer xValueFloatProp;
  if (node->GetProperty(xValueFloatProp, "X values")) {
    mitk::FloatLookupTable xValueLut = xValueFloatProp->GetValue();
    for (int i = 0; i < xValues.size(); ++i) {
      xValues[i] = xValueLut.GetTableValue(i);
    }
    newXValues = true;
  }
  else {
    for (int t = 0; t < xValues.size(); ++t) {
      xValues[t] = t;
    }
  }
  mitk::FloatLookupTableProperty::Pointer xValueOrderFloatProp;
  if (node->GetProperty(xValueOrderFloatProp, "X value order")) {
    mitk::FloatLookupTable xValueOrderLut = xValueOrderFloatProp->GetValue();
    for (int i = 0; i < xValueOrder.size(); ++i) {
      xValueOrder[i] = xValueOrderLut.GetTableValue(i);
    }
  }
  else {
    for (int t = 0; t < xValueOrder.size(); ++t) {
      xValueOrder[t] = t;
    }
    if (newXValues) {
      std::vector<double> xValues2 = xValues.toStdVector();
      std::vector<unsigned> xValueOrder2 = xValueOrder.toStdVector();
      MedianHybridQuickSort(xValues2, xValueOrder2);
      xValueOrder = QVector<unsigned>::fromStdVector(xValueOrder2);
    }
  }
}

void
IntensityProfileView::calculateRoiStatistics(mitk::DataNode* node, mitk::DataNode* roi) {
  Q_D(IntensityProfileView);

  mitk::Image::Pointer referenceImage = dynamic_cast<mitk::Image*>(node->GetData());
  unsigned timeSteps = referenceImage->GetTimeSteps();

  mitk::Image* roiImage = dynamic_cast<mitk::Image*>(roi->GetData());

  std::vector<Statistics> statisticsAtTimeSteps(timeSteps);
  if (roiImage->GetDimension() == 3) {
    for (unsigned t = 0; t < timeSteps; ++t) {
      mitk::ImageTimeSelector::Pointer timeSelector = mitk::ImageTimeSelector::New();
      timeSelector->SetInput(referenceImage);
      timeSelector->SetTimeNr(t);
      timeSelector->Update();
      mitk::Image::Pointer referenceImageAtTimeStep = timeSelector->GetOutput();

      mitk::ImageStatisticsCalculator::Pointer calculator = mitk::ImageStatisticsCalculator::New();
      calculator->SetImage(referenceImageAtTimeStep);
      calculator->SetImageMask(dynamic_cast<mitk::Image*>(roi->GetData()));
      calculator->SetMaskingMode(true);
      calculator->ComputeStatistics();
      statisticsAtTimeSteps[t] = calculator->GetStatistics();
    }
  }
  else { // if (roiDimension == 4) {
    mitk::ImageStatisticsCalculator::Pointer calculator = mitk::ImageStatisticsCalculator::New();
    calculator->SetImage(referenceImage);
    calculator->SetMaskingMode(true);
    for (unsigned t = 0; t < timeSteps; ++t) {
      calculator->SetImageMask(dynamic_cast<mitk::Image*>(roi->GetData()));
      calculator->SetMaskingMode(true);
      calculator->ComputeStatistics(t);
      statisticsAtTimeSteps[t] = calculator->GetStatistics(t);
    }
  }
  d->statisticsByNodeAndRoi[node][roi] = statisticsAtTimeSteps;
  d->statisticsByRoiAndNode[roi][node] = statisticsAtTimeSteps;
}

void IntensityProfileView::plotRoiProfiles(mitk::DataNode* node) {
  Q_D(IntensityProfileView);

  unsigned roiNumber = d->roiNodes.size();
  for (unsigned i = 0; i < roiNumber; ++i) {
    mitk::DataNode* roi = d->roiNodes[i];
    if (dimensionsAreEqual(node, roi, true)) {
      plotRoiProfile(node, roi);
    }
  }
}

void IntensityProfileView::plotRoiProfile(mitk::DataNode* node, mitk::DataNode* roi) {
  Q_D(IntensityProfileView);

  mitk::Image::Pointer referenceImage = dynamic_cast<mitk::Image*>(node->GetData());
  unsigned timeSteps = referenceImage->GetTimeSteps();

  int nodeSymbolIndex = d->referenceNodes.indexOf(node);
  if (nodeSymbolIndex == -1) {
    return;
  }
  const QwtSymbol& nodeSymbol = IntensityProfileViewPrivate::Symbols[nodeSymbolIndex];

  QVector<double> xValues(timeSteps);
  QVector<unsigned> xValueOrder(timeSteps);
  getXValues(node, xValues, xValueOrder);

  try {
    IntensityProfileViewPrivate::RoiStatisticsMap::iterator roiStatistics =
        d->statisticsByNodeAndRoi.find(node);
    if (roiStatistics == d->statisticsByNodeAndRoi.end() ||
        roiStatistics->find(roi) == roiStatistics->end()) {
      WaitCursorOn();
      calculateRoiStatistics(node, roi);
      WaitCursorOff();
    }

    QVector<double> xValuesOrdered(timeSteps);
    QVector<double> yValues(timeSteps);
    std::vector<Statistics> stats = d->statisticsByNodeAndRoi[node][roi];
    for (unsigned t = 0; t < timeSteps; ++t) {
      xValuesOrdered[t] = xValues[xValueOrder[t]];
      Statistics& stat = stats[xValueOrder[t]];
      yValues[t] = stat.Mean;
    }
    QString title = QString("%1 [%2]").
        arg(QString::fromStdString(node->GetName())).
        arg(QString::fromStdString(roi->GetName()));
    QwtPlotCurve* curve = new QwtPlotCurve(title);
    curve->setData(xValuesOrdered, yValues);
    float color[3];
    QwtSymbol roiSymbol = nodeSymbol;
    if (roi->GetColor(color)) {
      QPen pen = curve->pen();
      int red = static_cast<int>(color[0] * 255);
      int green = static_cast<int>(color[1] * 255);
      int blue = static_cast<int>(color[2] * 255);
      QColor qColor(red, green, blue);
      pen.setColor(qColor);
      curve->setPen(pen);
      roiSymbol.setPen(pen);
    }
    curve->setSymbol(roiSymbol);

    d->roiProfileMapsByNode[node][roi] = curve;
    curve->attach(ui->plotter);

    ui->plotter->replot();
  }
  catch (std::exception& exception) {
    WaitCursorOff();
    QMessageBox msgBox;
    msgBox.setWindowTitle("Error");
    msgBox.setText(tr("Error occurred during the calculation."));
    msgBox.setDetailedText(tr(exception.what()));
    msgBox.exec();
  }
}

void
IntensityProfileView::plotProfileNode(mitk::DataNode::Pointer node) {
  Q_D(IntensityProfileView);

//  mitk::LevelWindow levelWindow;
//  node->GetLevelWindow(levelWindow);
//  ui->plotter->setAxisScale(QwtPlot::yLeft, levelWindow.GetRangeMin(), levelWindow.GetRangeMax());

  mitk::FloatLookupTableProperty* profileProperty =
      dynamic_cast<mitk::FloatLookupTableProperty*>(node->GetProperty("intensity profile"));

  mitk::FloatLookupTable profile = profileProperty->GetValue();
  unsigned size = profile.GetLookupTable().size();

  QVector<double> xValues(size);
  QVector<unsigned> xValueOrder(size);
  getXValues(node, xValues, xValueOrder);
  QVector<double> xValuesOrdered(size);
  QVector<double> yValues(size);
  for (unsigned t = 0; t < size; ++t) {
    xValuesOrdered[t] = xValues[xValueOrder[t]];
    yValues[t] = profile.GetTableValue(xValueOrder[t]);
  }
  QwtPlotCurve* nodeProfile = new QwtPlotCurve(node->GetName().c_str());
  nodeProfile->setData(xValuesOrdered, yValues);
  float color[3];
  if (node->GetColor(color)) {
    QPen pen = nodeProfile->pen();
    QColor qColor((int)(color[0] * 255), (int)(color[1] * 255), (int)(color[2] * 255));
    pen.setColor(qColor);
    nodeProfile->setPen(pen);
  }
  bool isVisible;
  node->GetVisibility(isVisible, 0);
  if (isVisible) {
    nodeProfile->attach(ui->plotter);
  }
  d->storedProfileCurves.push_back(nodeProfile);

  ui->plotter->replot();
}

void
IntensityProfileView::plotStoredProfiles()
{
  Q_D(IntensityProfileView);

  for (int i = 0; i < d->profileNodes.size(); ++i) {
    mitk::DataNode* profileNode = d->profileNodes[i];
    plotProfileNode(profileNode);
  }
}

void
IntensityProfileView::onProfileNodeChanged()
{
  Q_D(IntensityProfileView);
  foreach (QwtPlotCurve* curve, d->storedProfileCurves) {
    delete curve;
  }
  d->storedProfileCurves.clear();
  plotStoredProfiles();
}

void
IntensityProfileView::on_storeCrosshairButton_clicked()
{
  Q_D(IntensityProfileView);

  static mitk::TNodePredicateDataType<mitk::PointSet>::Pointer hasPointSet =
      mitk::TNodePredicateDataType<mitk::PointSet>::New();

  mitk::IRenderWindowPart* display = GetRenderWindowPart();
  if (display != d->display) {
    if (d->display && d->showCrosshairProfile) {
      d->display->GetRenderWindow("transversal")->GetSliceNavigationController()->crosshairPositionEvent.RemoveListener(*d->crosshairPositionListener);
      d->display->GetRenderWindow("sagittal")->GetSliceNavigationController()->crosshairPositionEvent.RemoveListener(*d->crosshairPositionListener);
      d->display->GetRenderWindow("coronal")->GetSliceNavigationController()->crosshairPositionEvent.RemoveListener(*d->crosshairPositionListener);
    }
    d->display = display;
  }
  if (!display) {
    return;
  }
  const mitk::Point3D crosshairPosition = display->GetSelectedPosition();

  foreach (mitk::DataNode* node, d->referenceNodes) {
    mitk::PointSet::Pointer crosshairPoints;
    if (hasPointSet->CheckNode(node)) {
      crosshairPoints = dynamic_cast<mitk::PointSet*>(node->GetData());
    }
    else {
      crosshairPoints = mitk::PointSet::New();
      mitk::DataNode::Pointer crosshairNode = mitk::DataNode::New();
      QString profileName = QString("[%1; %2; %3]").
          arg(d->crosshairIndex[0]).
          arg(d->crosshairIndex[1]).
          arg(d->crosshairIndex[2]);
      crosshairNode->SetName(profileName.toStdString());
      crosshairNode->SetData(crosshairPoints);
      GetDataStorage()->Add(crosshairNode, node);
    }
    crosshairPoints->InsertPoint(crosshairPoints->GetSize(), crosshairPosition);
  }
}

void
IntensityProfileView::on_storeStatisticsButton_clicked()
{
//  Q_D(IntensityProfileView);

  // TODO add this feature back!

//  if (!d->showCrosshairProfile && !d->showRoiProfiles) {
//    return;
//  }
//
//  if (d->crosshairProfile) {
//    QString profileName = QString("[%1; %2; %3]").
//      arg(d->crosshairIndex[0]).
//      arg(d->crosshairIndex[1]).
//      arg(d->crosshairIndex[2]);
//
//    mitk::FloatLookupTable profile;
//    int dataSize = d->crosshairProfile->dataSize();
//    for (int i = 0; i < dataSize; ++i) {
//      profile.SetTableValue(i, d->crosshairProfile->y(i));
//    }
//    mitk::DataNode::Pointer profileNode = mitk::DataNode::New();
//    profileNode->SetName(profileName.toStdString());
//    profileNode->SetProperty("intensity profile", mitk::FloatLookupTableProperty::New(profile));
//    mitk::LevelWindow levelWindow;
//    d->referenceNode->GetLevelWindow(levelWindow);
//    profileNode->SetLevelWindow(levelWindow);
//    profileNode->SetVisibility(true);
//    float blue[] = {0.0, 0.0, 1.0};
//    profileNode->SetColor(blue);
//    GetDataStorage()->Add(profileNode, d->referenceNode);
//    plotProfileNode(profileNode);
//  }
//  if (d->showRoiProfiles) {
//    int timeSteps = d->statistics.size();
//
//    if (timeSteps == 0) {
//      return;
//    }
//    int roiNumber = d->statistics[0].size();
//    std::vector<mitk::FloatLookupTable> mean(roiNumber);
//    std::vector<mitk::FloatLookupTable> sigma(roiNumber);
//    std::vector<mitk::FloatLookupTable> rms(roiNumber);
//    std::vector<mitk::FloatLookupTable> min(roiNumber);
//    std::vector<mitk::FloatLookupTable> max(roiNumber);
//    std::vector<mitk::FloatLookupTable> n(roiNumber);
//
//    for (int t = 0; t < timeSteps; ++t) {
//      std::vector<Statistics> statisticsAtTimeStep = d->statistics[t];
//
//      for (unsigned i = 0; i < statisticsAtTimeStep.size(); ++i) {
//        const Statistics& roiStatisticsAtTimeStep = statisticsAtTimeStep[i];
//        mean[i].SetTableValue(t, roiStatisticsAtTimeStep.Mean);
//        sigma[i].SetTableValue(t, roiStatisticsAtTimeStep.Sigma);
//        rms[i].SetTableValue(t, roiStatisticsAtTimeStep.RMS);
//        min[i].SetTableValue(t, roiStatisticsAtTimeStep.Min);
//        max[i].SetTableValue(t, roiStatisticsAtTimeStep.Max);
//        n[i].SetTableValue(t, roiStatisticsAtTimeStep.N);
//      }
//    }
//    for (unsigned i = 0; i < roiNumber; ++i) {
//      mitk::DataNode::Pointer profileNode = mitk::DataNode::New();
//      profileNode->SetName(d->roiNodes[i]->GetName());
//      profileNode->SetProperty("intensity profile", mitk::FloatLookupTableProperty::New(mean[i]));
//      profileNode->SetProperty("statistics.mean", mitk::FloatLookupTableProperty::New(mean[i]));
//      profileNode->SetProperty("statistics.sigma", mitk::FloatLookupTableProperty::New(sigma[i]));
//      profileNode->SetProperty("statistics.RMS", mitk::FloatLookupTableProperty::New(rms[i]));
//      profileNode->SetProperty("statistics.min", mitk::FloatLookupTableProperty::New(min[i]));
//      profileNode->SetProperty("statistics.max", mitk::FloatLookupTableProperty::New(max[i]));
//      profileNode->SetProperty("statistics.N", mitk::FloatLookupTableProperty::New(n[i]));
//      float roiColor[3];
//      d->roiNodes[i]->GetColor(roiColor);
//      profileNode->SetColor(roiColor);
//      mitk::LevelWindow levelWindow;
//      d->referenceNode->GetLevelWindow(levelWindow);
//      profileNode->SetLevelWindow(levelWindow);
//      GetDataStorage()->Add(profileNode, d->referenceNode);
//    }
//  }
}

void
IntensityProfileView::on_copyStatisticsButton_clicked()
{
  Q_D(IntensityProfileView);

  if (!d->showCrosshairProfile && !d->showRoiProfiles) {
    QApplication::clipboard()->clear();
    return;
  }

  QString clipboard;
  foreach (mitk::DataNode* node, d->referenceNodes) {
    int timeSteps = dynamic_cast<mitk::Image*>(node->GetData())->GetTimeSteps();
    QVector<double> xValues(timeSteps);
    QVector<unsigned> xValueOrder(timeSteps);
    getXValues(node, xValues, xValueOrder);

    clipboard = clipboard.append(QString::fromStdString(node->GetName()));
    clipboard = clipboard.append("\n");
    if (!d->showRoiProfiles) {
      clipboard = clipboard.append("[%1; %2; %3]\n").
        arg(d->crosshairIndex[0]).
        arg(d->crosshairIndex[1]).
        arg(d->crosshairIndex[2]);
      clipboard = clipboard.append("Time step \t X value \t Intensity\n");
      for (int i = 0; i < timeSteps; ++i) {
        clipboard = clipboard.append( "%L1\t%L2\t%L3\n" ).arg(i).arg(xValues[i]).arg(d->crosshairProfiles[node]->y(i));
      }
    }
    else {
      int roiNumber = d->roiNodes.size();

      QString roiHeader(" ");
      QString columnHeader("Time step");
      roiHeader = roiHeader.append("\t");
      columnHeader = columnHeader.append("\t X value");
      if (d->showCrosshairProfile) {
        roiHeader = roiHeader.append("\t[%1; %2; %3]").
          arg(d->crosshairIndex[0]).
          arg(d->crosshairIndex[1]).
          arg(d->crosshairIndex[2]);
        columnHeader = columnHeader.append("\tIntensity");
      }
      for (int i = 0; i < roiNumber; ++i) {
        roiHeader = roiHeader.append("\t%1\t\t\t\t\t").arg(QString::fromStdString(d->roiNodes[i]->GetName()));
        columnHeader = columnHeader.append("\tMean\tStdDev\tRMS\tMin\tMax\tN");
      }
      clipboard = clipboard.append(roiHeader).append("\n");
      clipboard = clipboard.append(columnHeader).append("\n");

      for (int t = 0; t < timeSteps; ++t) {
        typedef QMap<mitk::DataNode*, std::vector<Statistics> > RoiStatistics;
        RoiStatistics roiStatistics = d->statisticsByNodeAndRoi[node];

        QString row;
        row = row.append("%L1").arg(t);
        row = row.append("\t%L1").arg(xValues[t]);
        if (d->showCrosshairProfile) {
          row = row.append("\t%L1").arg(d->crosshairProfiles[node]->y(t));
        }

        // iterate over the roi statistics
        RoiStatistics::iterator it = roiStatistics.begin();
        RoiStatistics::iterator end = roiStatistics.end();
        while (it != end) {
          const Statistics& roiStatisticsAtTimeStep = (*it)[t];
          // Copy statistics to clipboard ("%Ln" will use the default locale for
          // number formatting)
          row = row.append(" \t %L1 \t %L2 \t %L3 \t %L4 \t %L5 \t %L6")
            .arg(roiStatisticsAtTimeStep.Mean, 0, 'f', 10)
            .arg(roiStatisticsAtTimeStep.Sigma, 0, 'f', 10)
            .arg(roiStatisticsAtTimeStep.RMS, 0, 'f', 10)
            .arg(roiStatisticsAtTimeStep.Min, 0, 'f', 10)
            .arg(roiStatisticsAtTimeStep.Max, 0, 'f', 10)
            .arg(roiStatisticsAtTimeStep.N);
          ++it;
        }
        clipboard = clipboard.append(row).append("\n");
      }
    }
  }

  QApplication::clipboard()->setText(clipboard, QClipboard::Clipboard);
}

void
IntensityProfileView::on_clearCacheButton_clicked()
{
  Q_D(IntensityProfileView);

  typedef IntensityProfileViewPrivate::StatisticsAtTimeSteps StatisticsAtTimeSteps;
  typedef QMap<mitk::DataNode*, StatisticsAtTimeSteps> StatisticsMap;
  typedef QMap<mitk::DataNode*, QMap<mitk::DataNode*, StatisticsAtTimeSteps> > RoiStatisticsMap;

  foreach (StatisticsMap statisticsMap, d->statisticsByNodeAndRoi) {
    foreach (StatisticsAtTimeSteps statisticsAtTimeSteps, statisticsMap) {
      statisticsAtTimeSteps.clear();
    }
    statisticsMap.clear();
  }
  d->statisticsByNodeAndRoi.clear();

  foreach (StatisticsMap statisticsMap, d->statisticsByRoiAndNode) {
    statisticsMap.clear();
  }
  d->statisticsByRoiAndNode.clear();

  typedef QMap<mitk::DataNode*, QwtPlotCurve*> RoiProfiles;
  foreach (RoiProfiles roiProfiles, d->roiProfileMapsByNode) {
    foreach (QwtPlotCurve* profile, roiProfiles) {
      delete profile;
    }
    roiProfiles.clear();
  }
  d->roiProfileMapsByNode.clear();

}

bool
IntensityProfileView::eventFilter(QObject *obj, QEvent *event)
{
  Q_D(IntensityProfileView);
  if (obj == d->xAxis) {
    if (event->type() == QEvent::MouseButtonDblClick) {
      foreach (mitk::DataNode* node, d->referenceNodes) {
        bool hasChanged = askXValues(node);
        if (hasChanged) {
          deselectNode(node);
          selectNode(node);
        }
      }
      return true;
    }
    else if (event->type() == QEvent::Wheel) {
      // Mouse wheel events can be handled here. (E.g. to change the range.)
//      QWheelEvent* wheelEvent = dynamic_cast<QWheelEvent*>(event);
      return true;
    }
  }
  else if (obj == d->yAxis) {
    if (event->type() == QEvent::Wheel) {
        // Mouse wheel events can be handled here. (E.g. to change the range.)
//      QWheelEvent* wheelEvent = dynamic_cast<QWheelEvent*>(event);
      return true;
    }
  }
  // standard event processing
  return QObject::eventFilter(obj, event);
}


bool
IntensityProfileView::askXValues(mitk::DataNode* node)
{
  Q_D(IntensityProfileView);
  int timeSteps = dynamic_cast<mitk::Image*>(node->GetData())->GetDimension(3);
  std::vector<double> xValues(timeSteps);

  QString title = QString("Specify x values for %1").arg(QString::fromStdString(node->GetName()));
  QString text = QInputDialog::getText(
      d->m_Parent,
      title,
      "If the x values are stored as an image property, write\n"
      "the name of the property here.\n"
      "e.g.: TI, TR, FLIP_ANGLE\n"
      "\n"
      "If you want to use evenly spaced values, give the start\n"
      "and the end value of the interval, separated by a dash.\n"
      "e.g.: 1-6\n"
      "\n"
      "Otherwise, specify the values, separated by spaces.\n"
      "If you give less values than the number of time steps then\n"
      "the last interval will be used for the remaining elements.\n"
      "\n"
      "Type '0' to reset the values.\n");
  text = text.trimmed();
  if (text.isEmpty()) {
    return false;
  }

  mitk::FloatLookupTableProperty::Pointer xValueFloatProp;
  mitk::IntLookupTableProperty::Pointer xValueIntProp;
  mitk::StringLookupTableProperty::Pointer xValueStringProp;
  if (node->GetProperty(xValueFloatProp, text.toAscii().data())) {
    mitk::FloatLookupTable xValueLut = xValueFloatProp->GetValue();
    int n = xValueLut.GetLookupTable().size();
    for (int i = 0; i < n; ++i) {
      xValues[i] = xValueLut.GetTableValue(i);
    }
    if (n < timeSteps) {
      double interval = 1.0;
      double x = xValues[n - 1];
      if (n > 1) {
        interval = x - xValues[n - 2];
      }
      for (int i = n; i < timeSteps; ++i) {
        x += interval;
        xValues[i] = x;
      }
    }
  }
  else if (node->GetProperty(xValueIntProp, text.toAscii().data())) {
    mitk::IntLookupTable xValueLut = xValueIntProp->GetValue();
    int n = xValueLut.GetLookupTable().size();
    for (int i = 0; i < n; ++i) {
      xValues[i] = xValueLut.GetTableValue(i);
    }
    if (n < timeSteps) {
      double interval = 1.0;
      double x = xValues[n - 1];
      if (n > 1) {
        interval = x - xValues[n - 2];
      }
      for (int i = n; i < timeSteps; ++i) {
        x += interval;
        xValues[i] = x;
      }
    }
  }
  else if (node->GetProperty(xValueStringProp, text.toAscii().data())) {
    mitk::StringLookupTable xValueLut = xValueStringProp->GetValue();
    int n = xValueLut.GetLookupTable().size();
    for (int i = 0; i < n; ++i) {
      xValues[i] = std::atof(xValueLut.GetTableValue(i).c_str());
    }
    if (n < timeSteps) {
      double interval = 1.0;
      double x = xValues[n - 1];
      if (n > 1) {
        interval = x - xValues[n - 2];
      }
      for (int i = n; i < timeSteps; ++i) {
        x += interval;
        xValues[i] = x;
      }
    }
  }
  else if (text.contains('-')) {
    QStringList bounds = text.split('-', QString::SkipEmptyParts);
    if (bounds.size() != 2) {
      MITK_INFO << "Invalid interval format";
    }
    double a = bounds[0].toDouble();
    double b = bounds[1].toDouble();
    double delta = (b - a) / (timeSteps -1);
    for (int i = 0; i < timeSteps; ++i) {
      xValues[i] = a;
      a += delta;
    }
  }
  else {
    QStringList values = text.split(' ', QString::SkipEmptyParts);
    int i = 0, j = 0;
    bool ok = false;
    for (; i < values.size() && j < timeSteps; ++i) {
      double value = values[i].toDouble(&ok);
      if (ok) {
        xValues[j] = value;
        ++j;
      }
      else {
        break;
      }
    }
    if (ok) {
      if (j < timeSteps) {
        double interval = 1.0;
        double x = xValues[j - 1];
        if (j > 1) {
          interval = x - xValues[j - 2];
        }
        for (int i = j; i < timeSteps; ++i) {
          x += interval;
          xValues[i] = x;
        }
      }
    }
    else {
      return false;
    }
  }

  std::vector<unsigned> xValueOrder(timeSteps);
  MedianHybridQuickSort(xValues, xValueOrder);

  mitk::FloatLookupTable xValuesLut;
  mitk::FloatLookupTable xValueOrderLut;
  for (int i = 0; i < timeSteps; ++i) {
    xValuesLut.SetTableValue(i, static_cast<float>(xValues[i]));
    xValueOrderLut.SetTableValue(i, static_cast<float>(xValueOrder[i]));
  }

  node->SetProperty("X values", mitk::FloatLookupTableProperty::New(xValuesLut));
  node->SetProperty("X value order", mitk::FloatLookupTableProperty::New(xValueOrderLut));
  return true;
}

/*
 * Here is a modified sorting algorithm that re-makes the swapping
 * of the elements in a second array as well. Exactly the same
 * replacements are performed in the second array as in the first.
 * If you pass an array where the elements are equal to their indices
 * (0, 1, ...) as a second array then after the algorithm it will
 * store the order of the elements in the input array, before the
 * sorting.
 *
 *  Example:
 *  if the input arrays are these
 *     array: [3, 1, 9, 7], array2: [0, 1, 2, 3]
 *  then after the algorithm the arrays content will be
 *      array: [1, 3, 7, 9], array2: [1, 0, 3, 2]
 *
 *  Array2 expresses the order of elements in the first array
 *  before the sorting.
 *
 *  The algorithm is a modified version of quicksort that works
 *  fast for small number of elements. The code has been taken from
 *  here:
 *
 *  http://warp.povusers.org/SortComparison/
 */

void InsertionSort(std::vector<double>& array, std::vector<unsigned>& array2)
{
  unsigned size = array.size();
  for(unsigned i = 1; i < size; ++i)
  {
    double val = array[i];
    unsigned val2 = array2[i];
    unsigned j = i;
    while(j > 0 && val < array[j-1])
    {
      array[j] = array[j-1];
      array2[j] = array2[j-1];
      --j;
    }
    array[j] = val;
    array2[j] = val2;
  }
}

unsigned Partition(std::vector<double>& array, std::vector<unsigned>& array2, unsigned f, unsigned l, double pivot)
{
  unsigned i = f-1, j = l+1;
  while(true)
  {
    while(pivot < array[--j]);
    while(array[++i] < pivot);
    if(i<j)
    {
      double tmp = array[i];
      array[i] = array[j];
      array[j] = tmp;
      unsigned tmp2 = array2[i];
      array2[i] = array2[j];
      array2[j] = tmp2;
    }
    else
      return j;
  }
}

void MedianHybridQuickSortImpl(std::vector<double>& array, std::vector<unsigned>& array2, unsigned f, unsigned l)
{
  while(f+16 < l)
  {
    double v1 = array[f], v2 = array[l], v3 = array[(f+l)/2];
    double median =
        v1 < v2
        ? ( v3 < v1 ? v1 : std::min(v2, v3) )
        : ( v3 < v2 ? v2 : std::min(v1, v3) );
    unsigned m = Partition(array, array2, f, l, median);
    MedianHybridQuickSortImpl(array, array2, f, m);
    f = m+1;
  }
}

// Note that the first array is copied (passed by value). We want to know only the order
// of the elements, not to change it.
void MedianHybridQuickSort(std::vector<double> array, std::vector<unsigned>& array2)
{
  unsigned size = array.size();
  array2.resize(size);
  for (unsigned i = 0; i < size; ++i) {
    array2[i] = i;
  }
  MedianHybridQuickSortImpl(array, array2, 0, size - 1);
  InsertionSort(array, array2);
}

void
IntensityProfileView::setDefaultLevelWindow(mitk::DataNode* node)
{
  mitk::Image* image = dynamic_cast<mitk::Image*>(node->GetData());
  if (!image) {
    return;
  }

  double percentageOfRange = 90.0;
  double rangeMin = image->GetStatistics()->GetScalarValueMin();
  double rangeMax = image->GetStatistics()->GetScalarValueMax();
  double windowMin = rangeMin;
  double windowMax = rangeMin + (rangeMax - rangeMin) * percentageOfRange / 100.0;

  mitk::LevelWindow levelWindow;
  levelWindow.SetRangeMinMax(rangeMin, rangeMax);
  levelWindow.SetWindowBounds(windowMin, windowMax);
  node->SetLevelWindow(levelWindow);
}
