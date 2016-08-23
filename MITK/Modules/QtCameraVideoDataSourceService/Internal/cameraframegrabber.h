// Note: Taken from: http://omg-it.works/how-to-grab-video-frames-directly-from-qcamera/
// With thanks to the author(s).

#ifndef CAMERAFRAMEGRABBER_H
#define CAMERAFRAMEGRABBER_H

// Qt includes
#include <QAbstractVideoSurface>
#include <QList>

class CameraFrameGrabber : public QAbstractVideoSurface
{
    Q_OBJECT
public:
    explicit CameraFrameGrabber(QObject *parent = 0);

    QList<QVideoFrame::PixelFormat> supportedPixelFormats(QAbstractVideoBuffer::HandleType handleType) const;

    bool present(const QVideoFrame &frame);

signals:
    void frameAvailable(QImage frame);

public slots:

};

#endif // CAMERAFRAMEGRABBER_H

