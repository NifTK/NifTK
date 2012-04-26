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

#include "ImageInfoRenderer.h"

#include <itkSimpleDataObjectDecorator.h>

#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkCornerAnnotation.h>
#include <vtkTextProperty.h>

#include <mitkRenderingManager.h>
#include <mitkNodePredicateNot.h>
#include <mitkNodePredicateAnd.h>
#include <mitkNodePredicateData.h>
#include <mitkNodePredicateDataType.h>
#include <mitkNodePredicateProperty.h>
#include <mitkNodePredicateSource.h>
#include <mitkGenericProperty.h>
#include <mitkVtkLayerController.h>

#include <QString>
#include <QmitkRenderWindow.h>
#include <QmitkStdMultiWidget.h>

class ImageInfoRendererPrivate
{
public:
  mitk::DataStorage::Pointer dataStorage;
  mitk::RenderingManager* renderingManager;

  vtkRenderer* imageInfoRenderer[3];
  vtkCornerAnnotation* imageInfoAnnotation;

  QmitkRenderWindow* lastRenderWindow[3];

  QmitkStdMultiWidget* display;

  bool showCrosshairIntensity;
  bool pendingCrosshairPositionEvent;

  mitk::NodePredicateAnd::Pointer isCrosshairNode;
  mitk::NodePredicateAnd::Pointer pred;
};


ImageInfoRenderer::ImageInfoRenderer()
: d_ptr(new ImageInfoRendererPrivate)
{
  Q_D(ImageInfoRenderer);

  d->renderingManager = mitk::RenderingManager::GetInstance();
  d->dataStorage = GetDataStorage();

  d->imageInfoAnnotation = vtkCornerAnnotation::New();
  vtkTextProperty *textProp = vtkTextProperty::New();
  d->imageInfoAnnotation->SetMaximumFontSize(12);
  textProp->SetColor(1.0, 1.0, 1.0);
  textProp->SetBold(1);
   d->imageInfoAnnotation->SetTextProperty(textProp);
  for (int i = 0; i < 3; ++i) {
    d->imageInfoRenderer[i] = vtkRenderer::New();
    d->imageInfoRenderer[i]->AddActor(d->imageInfoAnnotation);
    d->lastRenderWindow[i] = 0;
  }

  d->display = 0;

  d->showCrosshairIntensity = true;
  d->pendingCrosshairPositionEvent = false;

  mitk::TNodePredicateDataType<mitk::Image>::Pointer isImage =
      mitk::TNodePredicateDataType<mitk::Image>::New();
  mitk::NodePredicateProperty::Pointer isVisible =
      mitk::NodePredicateProperty::New("visible", mitk::BoolProperty::New(true));
  mitk::NodePredicateProperty::Pointer isBinary =
      mitk::NodePredicateProperty::New("binary", mitk::BoolProperty::New(true));
  mitk::NodePredicateNot::Pointer isNotBinary =
      mitk::NodePredicateNot::New(isBinary);

  mitk::NodePredicateNot::Pointer isIncludedInBoundingBox =
      mitk::NodePredicateNot::New(
        mitk::NodePredicateProperty::New("includeInBoundingBox",
            mitk::BoolProperty::New(false)));

  d->isCrosshairNode =
      mitk::NodePredicateAnd::New(
          mitk::NodePredicateProperty::New("name", mitk::StringProperty::New("widget1Plane")),
          mitk::NodePredicateProperty::New("helper object", mitk::BoolProperty::New(true)));

  d->pred =
      mitk::NodePredicateAnd::New(
          isVisible,
          mitk::NodePredicateAnd::New(
              isImage,
              mitk::NodePredicateAnd::New(
                  isNotBinary,
                  isIncludedInBoundingBox)));
}

ImageInfoRenderer::~ImageInfoRenderer()
{
}

void
ImageInfoRenderer::onNodeAdded(const mitk::DataNode* node)
{
  if (node->IsVisible(0)) {
    onVisibilityChanged(node);
  }
}

void
ImageInfoRenderer::onNodeRemoved(const mitk::DataNode* node)
{
  if (node->IsVisible(0)) {
    onVisibilityChanged(node);
  }
}

void
ImageInfoRenderer::onVisibilityChanged(const mitk::DataNode* node)
{
  Q_D(ImageInfoRenderer);
  if (d->isCrosshairNode->CheckNode(node)) {
    onCrosshairVisibilityChanged(node);
  }
  else {
    if (d->showCrosshairIntensity) {

      // If there is a new StdMultiWidget, we have to reassign the crosshair position listeners.
      QmitkStdMultiWidget* display = GetActiveStdMultiWidget();
      if (d->display != display) {
        d->display = display;
        onCrosshairVisibilityChanged(node);
      }

      onCrosshairPositionEvent();
    }
  }
}

void
ImageInfoRenderer::onCrosshairVisibilityChanged(const mitk::DataNode* crosshairNode)
{
  Q_D(ImageInfoRenderer);
  d->showCrosshairIntensity = crosshairNode->IsVisible(0);;

  QmitkStdMultiWidget* display = GetActiveStdMultiWidget();

  if (display) {
    if (d->showCrosshairIntensity) {
      onCrosshairPositionEvent();
      display->GetRenderWindow1()->GetSliceNavigationController()->crosshairPositionEvent.AddListener(mitk::MessageDelegate<ImageInfoRenderer>(this, &ImageInfoRenderer::onCrosshairPositionEvent));
      display->GetRenderWindow2()->GetSliceNavigationController()->crosshairPositionEvent.AddListener(mitk::MessageDelegate<ImageInfoRenderer>(this, &ImageInfoRenderer::onCrosshairPositionEvent));
      display->GetRenderWindow3()->GetSliceNavigationController()->crosshairPositionEvent.AddListener(mitk::MessageDelegate<ImageInfoRenderer>(this, &ImageInfoRenderer::onCrosshairPositionEvent));
    }
    else {
      display->GetRenderWindow1()->GetSliceNavigationController()->crosshairPositionEvent.RemoveListener(mitk::MessageDelegate<ImageInfoRenderer>(this, &ImageInfoRenderer::onCrosshairPositionEvent));
      display->GetRenderWindow2()->GetSliceNavigationController()->crosshairPositionEvent.RemoveListener(mitk::MessageDelegate<ImageInfoRenderer>(this, &ImageInfoRenderer::onCrosshairPositionEvent));
      display->GetRenderWindow3()->GetSliceNavigationController()->crosshairPositionEvent.RemoveListener(mitk::MessageDelegate<ImageInfoRenderer>(this, &ImageInfoRenderer::onCrosshairPositionEvent));
      DisplayImageInfo("");
    }
  }
}

void
ImageInfoRenderer::onCrosshairPositionEvent()
{
  Q_D(ImageInfoRenderer);
  if (!d->pendingCrosshairPositionEvent) {
    d->pendingCrosshairPositionEvent = true;
    QTimer::singleShot(0, this, SLOT(onCrosshairPositionEventDelayed()));
  }
}

void
ImageInfoRenderer::onCrosshairPositionEventDelayed()
{
  Q_D(ImageInfoRenderer);
  d->pendingCrosshairPositionEvent = false;
  QmitkStdMultiWidget* display = GetActiveStdMultiWidget();
  if (display) {
    const mitk::Point3D crossPosition = display->GetCrossPosition();
    showPixelValue(crossPosition);
  }
}

void
ImageInfoRenderer::showPixelValue(mitk::Point3D crosshairPos)
{
  Q_D(ImageInfoRenderer);

  mitk::DataStorage::SetOfObjects::ConstPointer nodes =
      d->dataStorage->GetSubset(d->pred);

  mitk::DataStorage::SetOfObjects::const_iterator it = nodes->begin();
  mitk::DataStorage::SetOfObjects::const_iterator end = nodes->end();

  QString allText;

  while (it != end) {
    mitk::Image::Pointer image = dynamic_cast<mitk::Image*>((*it)->GetData());

    mitk::Index3D p;
    image->GetGeometry()->WorldToIndex(crosshairPos, p);
    unsigned timeStep = GetActiveStdMultiWidget()->GetRenderWindow1()->
        GetSliceNavigationController()->GetRenderer()->GetTimeStep();
    double pixelValue = image->GetPixelValueByIndex(p, timeStep);
    QString pixelValueText;
    if (pixelValue < 1.0 || pixelValue >= 10000.0) {
      pixelValueText = QString::number(pixelValue, 'e', 3);
    }
    else {
      pixelValueText = QString::number(pixelValue, 'f', 3);
    }
    QString text = QString("%1\nPixel value: %2").
        arg(QString::fromStdString((*it)->GetName())).
        arg(pixelValueText);
    allText += text;
    ++it;
    if (it != end) {
      allText += "\n";
    }
  }
  DisplayImageInfo(allText);
}

void
ImageInfoRenderer::DisplayImageInfo(const QString& text)
{
  for (int i = 0; i < 3; ++i) {
    DisplayImageInfo(text, i);
  }
}

void
ImageInfoRenderer::DisplayImageInfo(const QString& text, int i)
{
  Q_D(ImageInfoRenderer);
  QmitkStdMultiWidget* stdMultiWidget = GetActiveStdMultiWidget();
  if (!stdMultiWidget) {
    return;
  }
  QmitkRenderWindow* renderWindow;
  switch (i) {
  case 0: renderWindow = stdMultiWidget->GetRenderWindow1(); break;
  case 1: renderWindow = stdMultiWidget->GetRenderWindow2(); break;
  case 2: renderWindow = stdMultiWidget->GetRenderWindow3(); break;
  default: return;
  }

  if (d->lastRenderWindow[i] != renderWindow) {
    if (d->lastRenderWindow[i]) {
      QObject::disconnect(d->lastRenderWindow[i], SIGNAL(destroyed(QObject*)),
          this, SLOT(OnRenderWindowDelete(QObject*)));
    }
    d->lastRenderWindow[i] = renderWindow;
    if (d->lastRenderWindow[i]) {
      QObject::connect(d->lastRenderWindow[i], SIGNAL(destroyed(QObject*)),
          this, SLOT(OnRenderWindowDelete(QObject*)));
    }
  }

  if (d->lastRenderWindow[i]) {
    // …
    // add or remove text, “m_LastRenderWindow” is the mitk renderwindow you want to use
    mitk::VtkLayerController* lastRenderWindowLayerController =
        mitk::VtkLayerController::GetInstance(renderWindow->GetRenderWindow());
    if (!text.isEmpty())
    {
      d->imageInfoAnnotation->SetText(1, text.toLatin1().data());
      lastRenderWindowLayerController->InsertForegroundRenderer(d->imageInfoRenderer[i], true);
    }
    else if (lastRenderWindowLayerController->IsRendererInserted(d->imageInfoRenderer[i]))
    {
      lastRenderWindowLayerController->RemoveRenderer(d->imageInfoRenderer[i]);
    }
    d->lastRenderWindow[i]->update();
  }
}

void
ImageInfoRenderer::OnRenderWindowDelete(QObject * obj)
{
  Q_D(ImageInfoRenderer);
  for (int i = 0; i < 3; ++i) {
    if (obj == d->lastRenderWindow[i]) {
      d->lastRenderWindow[i] = 0;
    }
  }
}
