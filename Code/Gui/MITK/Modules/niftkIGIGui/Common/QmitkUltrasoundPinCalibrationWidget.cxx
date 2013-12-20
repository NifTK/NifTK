/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkUltrasoundPinCalibrationWidget.h"
#include <stdexcept>
#include <mitkOpenCVFileIOUtils.h>
#include <mitkFileIOUtils.h>
#include <niftkFileHelper.h>
#include <stdexcept>
#include <iostream>
#include <QApplication>
#include <QMessageBox>
#include <boost/regex.hpp>

//-----------------------------------------------------------------------------
QmitkUltrasoundPinCalibrationWidget::QmitkUltrasoundPinCalibrationWidget(
  const QString& inputTrackerDirectory,
  const QString& inputImageDirectory,
  const QString& outputMatrixDirectory,
  const QString& outputPointDirectory,  
  const unsigned long long timingToleranceInMilliseconds,
  QWidget *parent)
: QVTKWidget(parent)
, m_InputTrackerDirectory(inputTrackerDirectory)
, m_InputImageDirectory(inputImageDirectory)
, m_OutputMatrixDirectory(outputMatrixDirectory)
, m_OutputPointDirectory(outputPointDirectory)
, m_TimingToleranceInMilliseconds(timingToleranceInMilliseconds)
{
  m_ImageViewer = vtkImageViewer::New();
  this->SetRenderWindow(m_ImageViewer->GetRenderWindow());
  m_ImageViewer->SetupInteractor(this->GetRenderWindow()->GetInteractor());
  m_ImageViewer->SetColorLevel(127.5);
  m_ImageViewer->SetColorWindow(255);  
  
  m_PNGReader = vtkPNGReader::New();

  // Load all data, and set up the PNG reader to the first image.
  bool canFindTrackingData = mitk::CheckIfDirectoryContainsTrackingMatrices(m_InputTrackerDirectory.toStdString());
  if (!canFindTrackingData)
  {
    throw std::runtime_error("Could not find tracker matrices\n");
  }

  m_TrackingTimeStamps = mitk::FindTrackingTimeStamps(m_InputTrackerDirectory.toStdString());
  m_ImageFiles = niftk::FindFilesWithGivenExtension(m_InputImageDirectory.toStdString(), ".png");

  std::cout << "Found " << m_ImageFiles.size() << " image files..." << std::endl;
  std::cout << "Found " << m_TrackingTimeStamps.m_TimeStamps.size() << " tracking matrices..." << std::endl;

  m_ImageFileCounter = 0;
  m_PNGReader->SetFileName(m_ImageFiles[m_ImageFileCounter].c_str());
  m_ImageViewer->SetInputConnection(m_PNGReader->GetOutputPort());
  m_PNGReader->Update();

  int extent[6];
  m_PNGReader->GetDataExtent(extent);
  m_ImageWidth = extent[1] + 1;
  m_ImageHeight = extent[3] + 1;
}


//-----------------------------------------------------------------------------
QmitkUltrasoundPinCalibrationWidget::~QmitkUltrasoundPinCalibrationWidget()
{
}


//-----------------------------------------------------------------------------
void QmitkUltrasoundPinCalibrationWidget::enterEvent(QEvent* event)
{
  m_ImageViewer->GetRenderWindow()->SetSize(m_ImageWidth, m_ImageHeight);
  QVTKWidget::enterEvent(event);
}


//-----------------------------------------------------------------------------
void QmitkUltrasoundPinCalibrationWidget::mousePressEvent(QMouseEvent* event)
{
  try 
  {
    this->StorePoint(event);
    event->accept();
  }
  catch (std::exception& e)
  {
    std::cerr << "Caught std::exception:" << e.what();
    event->ignore();
  }
  catch (...)
  {
    std::cerr << "Caught unknown exception:";
    event->ignore();
  }
}


//-----------------------------------------------------------------------------
void QmitkUltrasoundPinCalibrationWidget::keyPressEvent(QKeyEvent* event)
{
  if (event->key() == Qt::Key_N)
  {
    this->NextImage();
    event->accept();    
  }
  else if (event->key() == Qt::Key_P)
  {
    this->PreviousImage();
    event->accept();
  }
  else if (event->key() == Qt::Key_Q)
  {
    this->QuitApplication();
    event->accept();
  }
  else
  {
    event->ignore();
  }
}


//-----------------------------------------------------------------------------
void QmitkUltrasoundPinCalibrationWidget::QuitApplication()
{
  QApplication::exit(0);
}


//-----------------------------------------------------------------------------
void QmitkUltrasoundPinCalibrationWidget::ShowImage(const unsigned long int& imageNumber)
{
  m_PNGReader->SetFileName(m_ImageFiles[imageNumber].c_str());
  m_ImageViewer->Render();
  std::cout << "Displaying image[" << imageNumber << "]=" << m_ImageFiles[imageNumber] << std::endl;
}


//-----------------------------------------------------------------------------
void QmitkUltrasoundPinCalibrationWidget::NextImage()
{
  if (m_ImageFileCounter < m_ImageFiles.size() - 1)
  {
    m_ImageFileCounter++;
    this->ShowImage(m_ImageFileCounter);
  }
}


//-----------------------------------------------------------------------------
void QmitkUltrasoundPinCalibrationWidget::PreviousImage()
{
  if (m_ImageFileCounter > 0)
  {
    m_ImageFileCounter--;
    this->ShowImage(m_ImageFileCounter);
  }
}


//-----------------------------------------------------------------------------
void QmitkUltrasoundPinCalibrationWidget::CreateDirectory(const std::string& dir)
{
  if (!niftk::DirectoryExists(dir))
  {
    if (!niftk::CreateDirectoryAndParents(dir))
    {
      QMessageBox::critical(this, tr("niftkUltrasoundPinCalibrationSorter"),
                                  tr("Can't write to\n%1").arg(QString::fromStdString(dir)),
                                  QMessageBox::Ok);
      QApplication::exit(-1);
    }
  }
}


//-----------------------------------------------------------------------------
void QmitkUltrasoundPinCalibrationWidget::StorePoint(QMouseEvent* event)
{
  if (event != NULL)
  {
    this->CreateDirectory(m_OutputPointDirectory.toStdString());
    this->CreateDirectory(m_OutputMatrixDirectory.toStdString());

    QString imageFileName = QString::fromStdString(m_ImageFiles[m_ImageFileCounter]);
    QRegExp rx("([0-9]{19})");

    int matchIndex = imageFileName.indexOf(rx);
    if (matchIndex != -1)
    {

      // Check if we have the right data.
      //
      // For a given image, where the filename is the timestamp,
      // we find the closest tracking matrix within tolerance.
      // If such a tracking matrix exists, we output both
      // matrix and point in separate files, named after the timestamp.
      QString timeStampString = imageFileName.mid(matchIndex,19);

      long long delta = 0;
      unsigned long long imageTimeStamp = timeStampString.toULongLong();
      m_TrackingTimeStamps.GetNearestTimeStamp(imageTimeStamp, &delta);

      if (fabs(delta) < m_TimingToleranceInMilliseconds)
      {
        // Output point.
        int xPixel = event->x();
        int yPixel = event->y();
        QString baseNameForPoint = timeStampString + QString(".txt");
        std::string pointFileFullPath = niftk::ConvertToFullNativePath((m_OutputPointDirectory + QString("/") + baseNameForPoint).toStdString());

        ofstream myfile(pointFileFullPath.c_str(), std::ofstream::out | std::ofstream::trunc);
        if (myfile.is_open())
        {
          myfile << xPixel << " " << yPixel << std::endl;
          myfile.close();
        }
        else
        {
          QMessageBox::warning(this, tr("niftkUltrasoundPinCalibrationSorter"),
                                      tr("Failed to write point to file\n%1").arg(QString::fromStdString(pointFileFullPath)),
                                      QMessageBox::Ok);
        }

        // Output matrix
        QString baseNameForMatrix = timeStampString + QString(".4x4");
        std::string matrixFileFullPath = niftk::ConvertToFullNativePath((m_OutputMatrixDirectory + QString("/") + baseNameForMatrix).toStdString());

        ofstream myfile2(matrixFileFullPath.c_str(), std::ofstream::out | std::ofstream::trunc);
        if (myfile2.is_open())
        {
          myfile2 << "Put matrix" << std::endl;
          myfile2.close();
        }
        else
        {
          QMessageBox::warning(this, tr("niftkUltrasoundPinCalibrationSorter"),
                                      tr("Failed to write matrix to file\n%1").arg(QString::fromStdString(matrixFileFullPath)),
                                      QMessageBox::Ok);
        }
      }
      else
      {
        QMessageBox::warning(this, tr("niftkUltrasoundPinCalibrationSorter"),
                                    tr("No tracking data for this image"),
                                    QMessageBox::Ok);
      }
    }
    else
    {
      QMessageBox::warning(this, tr("niftkUltrasoundPinCalibrationSorter"),
                                  tr("Invalid image file name\n%1").arg(imageFileName),
                                  QMessageBox::Ok);
    }
  }
}

