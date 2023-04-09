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
#include <unistd.h>

#include "zdraw.hh"


bool saveDoodles = false;
bool resloveDoodles = true;


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
    exportScaled();
    for (auto p: parts)
      delete p;
    parts.clear();
    points.clear();
  }
   QTimer::singleShot(std::chrono::milliseconds(100), this, &GraphicsView::onTimer);

}

static const char * classes[] = {
"cross", "dbackslash", "dslash", "heart", "house", "losange", "peak", "plus", "spiral", "square", "tornado", "tridown", "triup", "vlines", "waves"
};

void GraphicsView::exportScaled()
{
  static const int isz = 32;
  QGraphicsScene* scn = new QGraphicsScene();
  //compute extend
  auto l0 = parts.front()->line();
  double minx = std::min(l0.x1(), l0.x2());
  double miny = std::min(l0.y1(), l0.y2());
  double maxx = std::max(l0.x1(), l0.x2());
  double maxy = std::max(l0.y1(), l0.y2());
  for (auto l: parts)
  {
    auto ll = l->line();
    minx = std::min(minx, std::min(ll.x1(), ll.x2()));
    miny = std::min(miny, std::min(ll.y1(), ll.y2()));
    maxx = std::max(maxx, std::max(ll.x1(), ll.x2()));
    maxy = std::max(maxy, std::max(ll.y1(), ll.y2()));
  }
  float offx, offy, scale;
  if (maxx-minx > maxy-miny)
  {
    offx = minx;
    offy = miny - ((maxy-miny)-(maxx-minx))/2;
    scale = (double)isz / (maxx-minx);
  }
  else
  {
    offy = miny;
    offx = minx - ((maxx-minx)-(maxy-minx))/2;
    scale = (double)isz / (maxy-miny);
  }
  // inject scaled to scn
  for (auto l: parts)
  {
    auto ll = l->line();
    auto pen = QPen();
    pen.setWidth(2);
    scn->addLine((ll.x1()-offx)*scale, (ll.y1()-offy)*scale, (ll.x2()-offx)*scale, (ll.y2()-offy)*scale, pen);
  }
  
  QImage image(QSize(isz, isz), QImage::Format_RGB32);  // Create the image with the exact size of the shrunk scene
  image.fill(Qt::white);                                              // Start all pixels transparent
  QPainter painter(&image);
  scn->render(&painter);
  if (saveDoodles)
    image.save(("doodle-" + std::to_string(frameIndex++) + ".png").c_str());
  if (resloveDoodles)
  {
    image.save("/tmp/doodle.png");
    socket.writeDatagram("/tmp/doodle.png", strlen("/tmp/doodle.png"),
      QHostAddress("127.0.0.1"), 3333);
    char data[1024];
    unsigned long len;
    while (true)
    {
      std::cerr << "socket read" << std::endl;
      len = socket.readDatagram(data, 1024);
      if (len != -1)
        break;
      std::cerr << "socket bronk" << std::endl;
      usleep(1000);
    }
    data[len] = 0;
    std::cerr <<"net gave " << data <<std::endl;
    auto clsid = std::stoi(data);
    auto clsname = classes[clsid];
    if (text == nullptr)
    {
      text = scene()->addSimpleText("");
    }
    text->setText(clsname);
  }
  delete scn;
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
          auto pp = tp.lastPos();
          points.push_back(p);
          auto pen = QPen();
          pen.setWidth(SZ);
          parts.push_back(scene()->addLine(p.x(), p.y(), pp.x(), pp.y(), pen));
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