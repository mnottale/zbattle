#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsItem>
#include <QApplication>
#include <QTouchEvent>
#include <QWidget>
#include <QPixmap>
#include <QPushButton>
#include <QTabWidget>
#include <QSound>
#include <QTimer>
#include <QWindow>
#include <functional>
#include <memory>
#include <chrono>
#include <iostream>
#include <optional>

#include <math.h>
#include <unistd.h>
#include <vector>
#include <list>
#include <QUdpSocket>

enum class Symbol
{
  Cross,
  DSlash,
  DBackslash,
  Heart,
  House,
  Losange,
  Peak,
  Plus,
  Spiral,
  Square,
  Tornado,
  TriDown,
  TriUp,
  VLines,
  Waves,
};

enum class BuildingKind
{
  Nexus,
  Farm,
  Catapult,
};

template<typename T> class vector: public std::vector<T>
{
public:
  void swapOut(int idx)
  {
    std::swap((*this)[idx], (*this)[this->size()-1]);
    this->pop_back();
  }
};

template <typename T> class Grid
{
public:
  Grid(int w, int h, int nAxisSubdivide);
  void add(T const& o);
  void remove(T const& o);
  void move(T const& o, double x, double y);
  template<typename Cond>
  int countAround(double x, double y, double radius, Cond cond);
  template<typename Cond>
  vector<T> around(double x, double y, double radius, Cond cond);
  template<typename Cond>
  std::optional<T> closestAround(double x, double y, double radius, Cond cond);
private:
  vector<vector<T>> grid;
};

class Player;
class Game;

typedef std::chrono::high_resolution_clock::time_point Time;
Time now()
{
  return std::chrono::high_resolution_clock::now();
}

inline double norm2(QPointF p)
{
  return p.x()*p.x() + p.y()*p.y();
}

struct DrawContext
{
  void cleanup();
  Time lastTouch;
  vector<QGraphicsLineItem*> parts;
  vector<QPointF> points;
};

struct Zombie
{
  Player& owner;
  double px;
  double py;
  double hitpoints;
};
using ZombiePtr = std::shared_ptr<Zombie>;

using TouchCallback = std::function<void(QEvent::Type, QPointF)>;

struct TouchTarget
{
  double px;
  double py;
  double radius;
  TouchCallback onTouch;
};
using TouchTargePtr = std::shared_ptr<TouchTarget>;

struct Building
{
  Building(Player& owner)
  : owner(owner)
  {}
  Player& owner;
  BuildingKind kind;
  double px;
  double py;
  double radius;
  double hitpoints;
  QGraphicsPixmapItem* pixmap;
  TouchCallback onTouch;
  QGraphicsEllipseItem* handle = nullptr;
  QGraphicsEllipseItem* target = nullptr;
  TouchTargePtr handleTouch;
};

using BuildingPtr = std::shared_ptr<Building>;


class Player
{
public:
  Player(int index, Game& g, QRectF zone, bool flip);
  void onSymbol(int clsidx, QPointF location);
  void spawn(BuildingKind kind, QPointF center);
  void onHandle(BuildingPtr b, QEvent::Type t, QPointF p);
private:
  Game& game;
  int index;
  QRectF zone;
  bool flip;
  vector<BuildingPtr> buildings;
  vector<ZombiePtr> zombies;
  vector<TouchTargePtr> targets;
  friend class Game;
  friend class GraphicsView;
};

class Game
{
public:
  Game();
  void run(QApplication& app);
  void newDrawing(DrawContext const& ctx);
  void onNetworkTimer();
  void onSymbol(int clsid, QPointF center);
  QGraphicsScene scene;
  vector<Player> players;
  int w, h;
private:
  int nextIndex = 0;
  struct PendingDraw
  {
    PendingDraw(int pi, QPointF center)
    : sent(false)
    , pictureIndex(pi)
    , center(center)
    {
    }
    bool sent;
    int pictureIndex;
    QPointF center;
  };
  std::list<PendingDraw> requests;
  QUdpSocket socket;
};

class GraphicsView : public QGraphicsView
{
public:
    GraphicsView(Game& g, QGraphicsScene *scene = nullptr, QWidget *parent = nullptr);
    bool viewportEvent(QEvent *event) override;
    void onTimer();
private:
  Game& game;
  vector<DrawContext> contexts;
  Time lastTouch;
  int frameIndex = 0;
};


GraphicsView::GraphicsView(Game& g, QGraphicsScene *scene, QWidget *parent)
    : QGraphicsView(scene, parent)
    , game(g)
{
    viewport()->setAttribute(Qt::WA_AcceptTouchEvents);
    setDragMode(ScrollHandDrag);
    QTimer::singleShot(std::chrono::milliseconds(100), this, &GraphicsView::onTimer);
}
void GraphicsView::onTimer()
{
  for (int i=0; i< contexts.size(); i++)
  {
    if (now()-contexts[i].lastTouch > std::chrono::milliseconds(800))
    {
      game.newDrawing(contexts[i]);
      contexts[i].cleanup();
      contexts.swapOut(i);
      i--;
    }
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
        QTouchEvent *touchEvent = static_cast<QTouchEvent *>(event);
        QList<QTouchEvent::TouchPoint> touchPoints = touchEvent->touchPoints();
        for (auto tp: touchPoints)
        {
          auto p = tp.pos();
          auto pp = tp.lastPos();
          // priority 1: existing draw context
          DrawContext* hit = nullptr;
          for (auto& ctx: contexts)
          {
            if ((p-ctx.points.back()).manhattanLength() < 100)
            {
              hit = &ctx;
            }
          }
          if (hit == nullptr)
          {
            // priority2: touch targets
            for (auto& player: game.players)
            {
              for (auto& tgt: player.targets)
              {
                if (norm2(QPointF(tgt->px, tgt->py)-p) < tgt->radius*tgt->radius)
                {
                  if (tgt->onTouch)
                    tgt->onTouch(event->type(), p);
                  return true;
                }
              }
            }
            // priority3: buildings
            for (auto& player: game.players)
            {
              for (auto& tgt: player.buildings)
              {
                if (norm2(QPointF(tgt->px, tgt->py)-p) < tgt->radius*tgt->radius)
                {
                  if (tgt->onTouch)
                    tgt->onTouch(event->type(), p);
                  return true;
                }
              }
            }
            // no hit: new draw context
            contexts.emplace_back();
            hit = &contexts.back();
          }
          hit->points.push_back(p);
          auto pen = QPen();
          pen.setWidth(2);
          hit->parts.push_back(scene()->addLine(p.x(), p.y(), pp.x(), pp.y(), pen));
          hit->lastTouch = now();
        }
        return true;
    }
    default:
        break;
    }
    return QGraphicsView::viewportEvent(event);
}

void Game::newDrawing(DrawContext const& ctx)
{
  static const int isz = 32;
  QGraphicsScene* scn = new QGraphicsScene();
  //compute extend
  auto l0 = ctx.parts.front()->line();
  double minx = std::min(l0.x1(), l0.x2());
  double miny = std::min(l0.y1(), l0.y2());
  double maxx = std::max(l0.x1(), l0.x2());
  double maxy = std::max(l0.y1(), l0.y2());
  double sumx = 0, sumy = 0;
  for (auto l: ctx.parts)
  {
    auto ll = l->line();
    minx = std::min(minx, std::min(ll.x1(), ll.x2()));
    miny = std::min(miny, std::min(ll.y1(), ll.y2()));
    maxx = std::max(maxx, std::max(ll.x1(), ll.x2()));
    maxy = std::max(maxy, std::max(ll.y1(), ll.y2()));
    sumx += ll.x1()+ll.x2();
    sumy += ll.y1()+ll.y2();
  }
  sumx /= (double)ctx.parts.size()*2.0;
  sumy /= (double)ctx.parts.size()*2.0;
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
  for (auto l: ctx.parts)
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
  image.save(("doodle-" + std::to_string(nextIndex++) + ".png").c_str());
  requests.push_back(PendingDraw(nextIndex-1, QPointF(sumx, sumy)));
  // network task will take over from here
}

void DrawContext::cleanup()
{
  for (auto p: parts)
    delete p;
}

Player::Player(int index, Game& g, QRectF zone, bool flip)
: game(g)
, index(index)
, zone(zone)
, flip(flip)
{}

void Game::onNetworkTimer()
{
  QTimer::singleShot(std::chrono::milliseconds(100), &scene, [this] { onNetworkTimer();});
  if (requests.empty())
    return;
  auto&r = requests.front();
  if (!r.sent)
  {
    auto name = "doodle-" + std::to_string(r.pictureIndex) + ".png";
    socket.writeDatagram(name.c_str(), name.length(),
      QHostAddress("127.0.0.1"), 3333);
    r.sent = true;
  }
  if (!socket.hasPendingDatagrams())
    return;
  char data[1024];
  auto len = socket.readDatagram(data, 1024);
  if (len == -1)
  {
    std::cerr << "MEGABRONK SOCKET" << std::endl;
    return;
  }
  data[len] = 0;
  auto clsid = std::stoi(data);
  onSymbol(clsid, r.center);
  requests.pop_front();
}

Game::Game()
{
  QTimer::singleShot(std::chrono::milliseconds(100), &scene, [this] { onNetworkTimer();});
}

void Game::onSymbol(int clsid, QPointF center)
{
  qDebug() << "Game::onSymbol " << clsid << " " << center;
  for (auto& p: players)
  {
    if (p.zone.contains(center))
    {
      p.onSymbol(clsid, center);
      return;
    }
  }
}

void Player::onHandle(BuildingPtr b, QEvent::Type t, QPointF p)
{
  qDebug() << "do something " << t << " " << p;
  b->handle->setPos(p);
  b->handleTouch->px = p.x();
  b->handleTouch->py = p.y();
  auto anglerad = atan2(p.y()-b->py, p.x()-b->px);
  double angleDeg = anglerad * 180.0 / M_PI;
  b->pixmap->setRotation(angleDeg-90);
  if (t == QEvent::TouchEnd)
    game.scene.removeItem(b->target);
  else
  {
    game.scene.addItem(b->target);
    auto delta = p-QPointF(b->px, b->py);
    auto len = sqrt(norm2(delta));
    delta *= -1.5;
    b->target->setPos(QPointF(b->px, b->py) + delta);
    b->target->setRect(-len/2, -len/2, len, len);
  }
}

void Player::spawn(BuildingKind kind, QPointF center)
{
  static QPixmap pix_farm("assets/farm.png");
  static QPixmap pix_catapult("assets/catapult.png");
  auto b = std::make_shared<Building>(*this);
  b->kind = kind;
  b->px = center.x();
  b->py = center.y();
  b->radius = 150;
  b->hitpoints = 20;
  b->pixmap = game.scene.addPixmap(kind == BuildingKind::Farm ? pix_farm : pix_catapult);
  b->pixmap->setOffset(-100, -100);
  b->pixmap->setPos(b->px, b->py);
  if (kind == BuildingKind::Catapult)
  {
     b->handle = game.scene.addEllipse(-25, -25, 50, 50);
     b->handle->setPos(center.x()+100, center.y()+250);
     auto tt = std::make_shared<TouchTarget>();
     b->handleTouch = tt;
     auto sp = b->handle->scenePos();
     tt->px = sp.x();
     tt->py = sp.y();
     tt->radius = 25;
     tt->onTouch = [this, b](QEvent::Type t, QPointF p) { onHandle(b, t, p);};
     b->target = new QGraphicsEllipseItem(-25, -25, 50, 50);
     b->target->setPen(QPen(Qt::red));
     targets.push_back(tt);
  }
  buildings.push_back(b);
}

void Player::onSymbol(int clsid, QPointF center)
{
  qDebug() << "Player::onSymbol " << clsid << " " << center;
  Symbol s = (Symbol)clsid;
  if (s == Symbol::TriUp)
    spawn(BuildingKind::Catapult, center);
  else if (s == Symbol::House)
    spawn(BuildingKind::Farm, center);
  else if (s == Symbol::Cross)
  {
    for (int i=0; i<buildings.size(); i++)
    {
      auto& b = *buildings[i];
      if (norm2(center-QPointF(b.px, b.py)) < b.radius*b.radius)
      {
        delete b.pixmap;
        buildings.swapOut(i);
        break;
      }
    }
  }
}

void Game::run(QApplication& app)
{
  w = 3840;
  h = 2160;
  float scale = 1;
  GraphicsView* view = new GraphicsView(*this, &scene);
  view->setSceneRect(0,0,w, h);
  //view.scale(scale,scale);
  view->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  view->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  view->show();
  view->windowHandle()->setPosition(0, 0);
  view->resize(w*scale+1, h*scale+1);
  view->showFullScreen();
  players.reserve(2);
  players.emplace_back(0, *this, QRectF(0, 0, w/2, h), false);
  players.emplace_back(1, *this, QRectF(w/2, 0, w, h), false);
  app.exec();
}

int main(int argc, char **argv)
{
  QApplication app(argc, argv);
  Game* g = new Game();
  g->run(app);
}