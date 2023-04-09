#include <QGraphicsView>
#include <vector>
#include <QUdpSocket>

typedef std::chrono::high_resolution_clock::time_point Time;
Time now();
class GraphicsView : public QGraphicsView
{

public:
    GraphicsView(QGraphicsScene *scene = nullptr, QWidget *parent = nullptr);

    bool viewportEvent(QEvent *event) override;
    void onTimer();
    void exportScaled();
private:
  QUdpSocket socket;
  std::vector<QGraphicsLineItem*> parts;
  std::vector<QPointF> points;
  QGraphicsSimpleTextItem* text = nullptr;
  Time lastTouch;
  int frameIndex = 0;
};