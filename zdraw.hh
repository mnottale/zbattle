#include <QGraphicsView>
#include <vector>

typedef std::chrono::high_resolution_clock::time_point Time;
Time now();
class GraphicsView : public QGraphicsView
{

public:
    GraphicsView(QGraphicsScene *scene = nullptr, QWidget *parent = nullptr);

    bool viewportEvent(QEvent *event) override;
    void onTimer();
private:
  std::vector<QGraphicsEllipseItem*> parts;
  std::vector<QPointF> points;
  Time lastTouch;
};