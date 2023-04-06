#include <QGraphicsScene>
#include <QGraphicsView>
#include <QGraphicsItem>
#include <QApplication>
#include <QTouchEvent>
#include <QWidget>
#include <QPixmap>
#include <QPushButton>
#include <QTabWidget>
#include <QSound>
#include <QTimer>
#include <functional>
#include <memory>
#include <chrono>
#include <iostream>

#include <math.h>

#include "zdraw.hh"





Time now()
{
  return std::chrono::high_resolution_clock::now();
}

GraphicsView::GraphicsView(QGraphicsScene *scene, QWidget *parent)
    : QGraphicsView(scene, parent)
{
    viewport()->setAttribute(Qt::WA_AcceptTouchEvents);
    setDragMode(ScrollHandDrag);
    QTimer::singleShot(std::chrono::milliseconds(100), this, &GraphicsView::onTimer);
}
static const int SZ = 2;
static const int HSZ = SZ / 2;

void GraphicsView::onTimer()
{
  if (now()-lastTouch > std::chrono::milliseconds(1000) && points.size() > 0)
  {
    for (auto p: parts)
      delete p;
    parts.clear();
    points.clear();
  }
   QTimer::singleShot(std::chrono::milliseconds(100), this, &GraphicsView::onTimer);

}

bool GraphicsView::viewportEvent(QEvent *event)
{
    switch (event->type()) {
    case QEvent::TouchBegin:
    case QEvent::TouchUpdate:
    case QEvent::TouchEnd:
    {
        lastTouch = now();
        QTouchEvent *touchEvent = static_cast<QTouchEvent *>(event);
        QList<QTouchEvent::TouchPoint> touchPoints = touchEvent->touchPoints();
        for (auto tp: touchPoints)
        {
          auto p = tp.pos();
          points.push_back(p);
          parts.push_back(scene()->addEllipse(p.x()-HSZ, p.y()-HSZ,SZ,SZ, QPen(), QBrush()));
        }
        /*
        if (touchPoints.count() == 2) {
            // determine scale factor
            const QTouchEvent::TouchPoint &touchPoint0 = touchPoints.first();
            const QTouchEvent::TouchPoint &touchPoint1 = touchPoints.last();
            qreal currentScaleFactor =
                    QLineF(touchPoint0.pos(), touchPoint1.pos()).length()
                    / QLineF(touchPoint0.startPos(), touchPoint1.startPos()).length();
            if (touchEvent->touchPointStates() & Qt::TouchPointReleased) {
                // if one of the fingers is released, remember the current scale
                // factor so that adding another finger later will continue zooming
                // by adding new scale factor to the existing remembered value.
                totalScaleFactor *= currentScaleFactor;
                currentScaleFactor = 1;
            }
            setTransform(QTransform::fromScale(totalScaleFactor * currentScaleFactor,
                                               totalScaleFactor * currentScaleFactor));
        }
        */
        return true;
    }
    default:
        break;
    }
    return QGraphicsView::viewportEvent(event);
}

int main(int argc, char **argv)
{
  QApplication app(argc, argv);
  QGraphicsScene* scene= new QGraphicsScene();
  GraphicsView* view = new GraphicsView(scene);
  view->setSceneRect(0,0,1024,1024);
  view->show();
  app.exec();
}