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
#include <algorithm>
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

static const unsigned int N_SYMBOLS = 15;

const char* symbolNames[] = {
  "cross", "dslash", "dbackslash", "heart", "house", "losange", "peak", "plus",
  "spiral", "square", "tornado", "tridown", "triup", "vlines", "waves"
};


enum class BuildingKind
{
  Farm,
  Catapult,
};

const char* symbolHelp[] = {
   "destroy one of your buildings",
   "",
   "",
   "",
   "",
   "",
   "shockwave: send Zs flying around",
   "spawn burst: spawn a burst of Zs on one of your farms",
   "",
   "farm: build a farm that spawns Zs regularly",
   "",
   "",
   "catapult: build a catapult to send smoke signals that will attract your Zs",
   "",
   "stealth: make one of your buildings invisible to ennemy Zs for a short time",
};

namespace C
{
  const double buildingHitpoints[] = {200, 50};
  const int symbolCooldownsMs[N_SYMBOLS] = {
    0, //cross
    -1,
    -1,
    -1, // heart
    -1, // house
    -1, // losange
    10000, // peak
    15000, // plus
    -1, // spiral
    30000, // square
    -1,  // tornado
    -1, // tridown
    8000, // triup
    -1, // vlines
    30000, // waves
  };
  const double symbolsManaCost[N_SYMBOLS] = {
    0, //cross
    -1,
    -1,
    -1, // heart
    -1, // house
    -1, // losange
    10, // peak
    40, // plus
    -1, // spiral
    50, // square
    -1,  // tornado
    -1, // tridown
    30, // triup
    -1, // vlines
    40, // waves
  };
  const std::vector<int> activeSymbols = {0, 6,7,9,12,14};
  const int farmsTotal = 6;
  const int farmsLiveMax = 4;
  const double manaMax = 100;
  const double manaRegenRate = 1.0;
  const int zombieSpawnIntervalMs = 2000;
  const double zombieMoveSpeed = 100;
  const double buildingWeight = 1;
  const double smokeWeight = 4;
  const int smokeLifeTimeMs = 20000;
  const double zombieAggroDistance = 200;
  const double zombieCombatDistance = 8.0;
  const double zombieSupportDistance = 10.0;
  const double zombieOverwhelmingSupport = 10.0; // number of z for max support effect
  const double zombieBaseDmg = 0.7;
  const double zombieExtraMaxDmg = 0.6;
  const double zombieStructureDmg = 1.0;
  const double zombieHitpoints = 1;
  const double waveRadius = 400;
  const double waveDisplacmentFactorMin = 1.2;
  const double waveDisplacmentFactorMax = 2.5;
  const int burstZombieCount = 10;
  const int hideDurationMs = 8000;
  const double zombieSpawnSpread = 40;
  const double zombieExplosionRadius = 8.0;
  const double zombieExplosionDamage = 0.07;
}

inline double notZero(double v)
{
  if (std::abs(v) < 1e-8)
    return 1e-8;
  return v;
}
template<typename T> class vector: public std::vector<T>
{
public:
  void swapOut(int idx)
  {
    std::swap((*this)[idx], (*this)[this->size()-1]);
    this->pop_back();
  }
  void swapOut(T const& obj)
  {
    auto it = std::find(this->begin(), this->end(), obj);
    if (it == this->end())
      return;
    auto idx = it - this->begin();
    swapOut(idx);
  }
};

template <typename T> class Grid
{
public:
  Grid(int w, int h, double zoneSize);
  void add(T const& o);
  void remove(T const& o);
  void move(T const& o, double x, double y);
  template<typename Cond>
  int countAround(double x, double y, double radius, Cond cond);
  template<typename Cond>
  vector<T> around(double x, double y, double radius, Cond cond);
  template<typename Cond>
  vector<T> aroundWide(double x, double y, double radius, Cond cond);
  template<typename Cond>
  std::optional<T> closestAround(double x, double y, double radius, Cond cond);
private:
  vector<T>& at(double x, double y, int ox=0, int oy=0);
  int sx, sy, w, h;
  double stride;
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
struct Zombie;
using ZombiePtr = std::shared_ptr<Zombie>;
struct Zombie
{
  Zombie(Player& o):owner(o){}
  Player& owner;
  double px;
  double py;
  double hitpoints;
  QGraphicsEllipseItem* pixmap;
};

inline QPointF position(ZombiePtr const& z)
{
  return QPointF(z->px, z->py);
}




template<typename T>
Grid<T>::Grid(int w, int h, double zoneSize)
:w(w), h(h), stride(zoneSize)
{
  sx = ceil((double)w / zoneSize);
  sy = ceil((double)h / zoneSize);
  grid.resize(sx*sy);
}
template<typename T>
vector<T>&
Grid<T>::at(double x, double y, int ox, int oy)
{
  static vector<T> empty;
  int cx = floor(x/stride);
  int cy = floor(y/stride);
  cx += ox;
  cy += oy;
  if (cx<0 || cy<0 || cx >= sx || cy >= sy)
    return empty;
  return grid[cy*sx+cx];
}

template<typename T>
void Grid<T>::add(T const& o)
{
  auto p = position(o);
  at(p.x(), p.y()).push_back(o);
}
template<typename T>
void Grid<T>::remove(T const& o)
{
  auto p = position(o);
  at(p.x(), p.y()).swapOut(o);
}
template<typename T>
void Grid<T>::move(T const& o, double x, double y)
{
  auto pprev = position(o);
  auto& vprev = at(pprev.x(), pprev.y());
  auto& vnew = at(x, y);
  if (&vnew != &vprev)
  {
    vprev.swapOut(o);
    vnew.push_back(o);
  }
}
template<typename T>
template<typename Cond>
int Grid<T>::countAround(double x, double y, double radius, Cond cond)
{
  int result = 0;
  auto consider = [&](vector<T>& v)
  {
    for (auto& o: v)
    {
      if (norm2(QPointF(x, y)-position(o)) < radius*radius && cond(o))
        result++;
    }
  };
  auto& v = at(x, y);
  consider(v);
  double xs = x/stride;
  double ys = x/stride;
  bool xneg =  (xs-floor(xs) <= ceil(xs)-xs);
  bool yneg = (ys-floor(ys) <= ceil(ys)-ys);
  consider(at(x,y, xneg? -1:1, 0));
  consider(at(x, y, 0, yneg? -1:1));
  consider(at(x,y, xneg? -1:1, yneg? -1:1));
  return result;
}
template<typename T>
template<typename Cond>
vector<T> Grid<T>::around(double x, double y, double radius, Cond cond)
{
  vector<T> result;
  auto consider = [&](vector<T>& v)
  {
    for (auto& o: v)
    {
      if (norm2(QPointF(x, y)-position(o)) < radius*radius && cond(o))
        result.push_back(o);
    }
  };
  auto& v = at(x, y);
  consider(v);
  double xs = x/stride;
  double ys = x/stride;
  bool xneg =  (xs-floor(xs) <= ceil(xs)-xs);
  bool yneg = (ys-floor(ys) <= ceil(ys)-ys);
  consider(at(x,y, xneg? -1:1, 0));
  consider(at(x, y, 0, yneg? -1:1));
  consider(at(x,y, xneg? -1:1, yneg? -1:1));
  return result;
}
template<typename T>
template<typename Cond>
std::optional<T> Grid<T>::closestAround(double x, double y, double radius, Cond cond)
{
  std::optional<T> result;
  auto consider = [&](vector<T>& v)
  {
    for (auto& o: v)
    {
      auto n2 = norm2(QPointF(x, y)-position(o));
      if (n2 < radius*radius && cond(o)
        && (!result || norm2(QPointF(x, y)-position(*result)) > n2))
        result = o;
    }
  };
  auto& v = at(x, y);
  consider(v);
  double xs = x/stride;
  double ys = x/stride;
  bool xneg =  (xs-floor(xs) <= ceil(xs)-xs);
  bool yneg = (ys-floor(ys) <= ceil(ys)-ys);
  consider(at(x,y, xneg? -1:1, 0));
  consider(at(x, y, 0, yneg? -1:1));
  consider(at(x,y, xneg? -1:1, yneg? -1:1));
  return result;
}
template<typename T>
template<typename Cond>
vector<T> Grid<T>::aroundWide(double x, double y, double radius, Cond cond)
{
  vector<T> result;
  auto consider = [&](vector<T>& v)
  {
    for (auto& o: v)
    {
      if (norm2(QPointF(x, y)-position(o)) < radius*radius && cond(o))
        result.push_back(o);
    }
  };
  int minx = floor((x-radius)/stride);
  int maxx = floor((x+radius)/stride);
  int miny = floor((y-radius)/stride);
  int maxy = floor((y+radius)/stride);
  minx = std::max(minx, 0);
  miny = std::max(miny, 0);
  maxx = std::min(sx-1, maxx);
  maxy = std::min(sy-1, maxy);
  for (int cx=minx; cx<=maxx; cx++)
  {
    for (int cy=miny; cy<=maxy; cy++)
      consider(grid[cx+cy*sx]);
  }
  return result;
}

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
  QGraphicsRectItem* hpBar;
  // catapult stuff
  TouchCallback onTouch;
  QGraphicsEllipseItem* handle = nullptr;
  QGraphicsEllipseItem* target = nullptr;
  TouchTargePtr handleTouch;
  // farm stuff
  Time lastSpawnTime;
  Time hiddenUntil;
};

using BuildingPtr = std::shared_ptr<Building>;

class ManagedAnimation;
struct Smoke
{
  double px;
  double py;
  Time createdAt;
  std::unique_ptr<ManagedAnimation> animation;
};
using SmokePtr = std::shared_ptr<Smoke>;

class Player
{
public:
  Player(int index, Game& g, QRectF zone, int rotation);
  void onSymbol(int clsidx, QPointF location);
  void spawn(BuildingKind kind, QPointF center);
  void onHandle(BuildingPtr b, QEvent::Type t, QPointF p);
  void fire(BuildingPtr b, QPointF p);
  void tick();
  void spawnZombie(BuildingPtr where);
  void onAreaClick(QPointF p);
  bool operator == (const Player& b) const { return index == b.index;}
  bool operator != (const Player& b) const { return index != b.index;}
private:
  Game& game;
  int index;
  QRectF zone;
  int rotation;
  vector<BuildingPtr> buildings;
  vector<ZombiePtr> zombies;
  vector<TouchTargePtr> targets;
  vector<SmokePtr> smokes;
  Time lastTick;
  double mana;
  int farmsRemaining;
  Time symbolLastUsedTime[N_SYMBOLS];
  QGraphicsRectItem* cooldownRect[N_SYMBOLS];
  QGraphicsRectItem* area;
  QGraphicsRectItem* manaRect;
  QGraphicsTextItem* helpText;
  QGraphicsTextItem* farmsCounter;
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
  void tick();
  void move(ZombiePtr& z, QPointF pos);
  QGraphicsScene scene;
  vector<Player> players;
  int w, h;
  Grid<ZombiePtr>* grid;

  //assets
  std::vector<QPixmap*> smoke[4];
  std::vector<QPixmap*> symbols;
  QColor pColors[4] = {QColor(255,0,255), QColor(0,255,255), QColor(255,255,0), QColor(255,0,0)};
private:
  Time lastTick;
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
  Time lastPlayerTick;
};

class ManagedAnimation: public QGraphicsPixmapItem
{
public:
  ManagedAnimation(std::vector<QPixmap*> const& assets, int frameTimeMs, double scale)
  :QGraphicsPixmapItem(*assets[0])
  , _a(assets)
  {
    setScale(scale);
    _timer = new QTimer();
    _timer->connect(_timer, &QTimer::timeout, std::bind(&ManagedAnimation::onTimer, this));
    _timer->setInterval(frameTimeMs);
    _timer->start();
  }
  ~ManagedAnimation()
  {
    delete _timer;
  }
  void onTimer()
  {
    ++_idx;
    if (_idx >= _a.size())
    {
      _idx = 0;
      /*
      _timer->stop();
      delete _timer;
      _game.scene().removeItem(this);
      delete this;
      return;*/
    }
    setPixmap(*_a[_idx]);
  }
private:
  const std::vector<QPixmap*>& _a;
  QTimer* _timer;
  int _idx;
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
          // priority0: touch targets
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
          // priority 1: existing draw context
          DrawContext* hit = nullptr;
          for (auto& ctx: contexts)
          {
            if ((p-ctx.points.back()).manhattanLength() < 700)
            {
              hit = &ctx;
            }
          }
          if (hit == nullptr)
          {
            // zone target
            if (p.x() < 128+50)
            {
              game.players[0].onAreaClick(p);
              return true;
            }
            if (p.x() > game.w-128-50)
            {
              game.players[1].onAreaClick(p);
              return true;
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
          pen.setColor(Qt::red);
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

Player::Player(int index, Game& g, QRectF zone, int rotation)
: game(g)
, index(index)
, zone(zone)
, rotation(rotation)
{
  const int H = 128 + 50;
  mana = C::manaMax;
  farmsRemaining = C::farmsTotal;
  area = game.scene.addRect(0, 0, game.w, H);
  area->setRotation(rotation);
  if (rotation == 90)
    area->setPos(H, 0);
  else if (rotation == 270)
    area->setPos(game.w-H, game.h);
  manaRect = new QGraphicsRectItem(0,0,game.h, 50);
  manaRect->setBrush(QBrush(Qt::red));
  manaRect->setParentItem(area);
  manaRect->setPos(0, H-50);
  int offset = 0;
  for (auto i: C::activeSymbols)
  {
    auto* q = new QGraphicsPixmapItem(*game.symbols[i]);
    q->setParentItem(area);
    q->setScale(4.0);
    q->setPos(offset*128, 0);
    auto* cd = new QGraphicsRectItem(0,0,32,32);
    cd->setBrush(QBrush(QColor(0,255,0,30)));
    cd->setParentItem(q);
    cooldownRect[i] = cd;
    offset++;
    if (i == (int)Symbol::Square)
    {
      farmsCounter = new QGraphicsTextItem(std::to_string(farmsRemaining).c_str());
      farmsCounter->setParentItem(cd);
    }
  }
  helpText = new QGraphicsTextItem("here is what you can draw, click for help");
  helpText->setDefaultTextColor(Qt::red);
  helpText->setParentItem(area);
  helpText->setPos(0, -50);
  lastTick = now();
}

void Player::onAreaClick(QPointF p)
{
  int localX = p.y();
  if (rotation == 270)
    localX = game.h-localX;
  int offset = localX / 128;
  if (offset < C::activeSymbols.size())
  {
    auto* text = symbolHelp[C::activeSymbols[offset]];
    helpText->setPlainText(text);
  }
}

void Game::onNetworkTimer()
{
  QTimer::singleShot(std::chrono::milliseconds(100), &scene, [this] { onNetworkTimer();});
  if (requests.empty())
    return;
  auto&r = requests.front();
  if (!r.sent)
  {
    auto name = "doodle-" + std::to_string(r.pictureIndex) + ".png";
    int rotation = 0;
    for (auto& p: players)
    {
      if (p.zone.contains(r.center))
      {
        rotation = p.rotation;
        break;
      }
    }
    name += " " + std::to_string(rotation);
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

void Player::fire(BuildingPtr b, QPointF p)
{
  auto delta = p-QPointF(b->px, b->py);
  //auto len = sqrt(norm2(delta));
  delta *= -2.5;
  auto spos = QPointF(b->px, b->py) + delta;
  auto smoke = std::make_shared<Smoke>();
  smoke->px = spos.x();
  smoke->py = spos.y();
  smoke->createdAt = now();
  smoke->animation = std::make_unique<ManagedAnimation>(
    game.smoke[index], 100, 1.0);
  game.scene.addItem(&*smoke->animation);
  smoke->animation->setOffset(-607/2, -524/2);
  smoke->animation->setPos(spos);
  smokes.push_back(smoke);
}

void Player::onHandle(BuildingPtr b, QEvent::Type t, QPointF p)
{
  b->handle->setPos(p);
  b->handleTouch->px = p.x();
  b->handleTouch->py = p.y();
  auto anglerad = atan2(p.y()-b->py, p.x()-b->px);
  double angleDeg = anglerad * 180.0 / M_PI;
  b->pixmap->setRotation(angleDeg-90);
  if (t == QEvent::TouchEnd)
  {
    auto delta = p-QPointF(b->px, b->py);
    auto len = sqrt(norm2(delta));
    if (len > 50)
      fire(b, p);
    delta *= 250.0 / len;
    auto res = delta + QPointF(b->px, b->py);
    b->handle->setPos(res);
    b->handleTouch->px = res.x();
    b->handleTouch->py = res.y();
    game.scene.removeItem(b->target);
  }
  else
  {
    game.scene.addItem(b->target);
    auto delta = p-QPointF(b->px, b->py);
    auto len = sqrt(norm2(delta));
    delta *= -2.5;
    b->target->setPos(QPointF(b->px, b->py) + delta);
    b->target->setRect(-len/4, -len/4, len, len);
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
  b->radius = 100;
  b->hitpoints = C::buildingHitpoints[(int)kind];
  b->pixmap = game.scene.addPixmap(kind == BuildingKind::Farm ? pix_farm : pix_catapult);
  b->pixmap->setOffset(-100, -100);
  b->pixmap->setPos(b->px, b->py);
  b->pixmap->setRotation(rotation);
  b->pixmap->setZValue(4);
  b->hpBar = new QGraphicsRectItem(-100,0,200,25);
  b->hpBar->setBrush(QBrush(Qt::red));
  b->hpBar->setZValue(5);
  b->hpBar->setParentItem(b->pixmap);
  if (kind == BuildingKind::Catapult)
  {
    b->target = new QGraphicsEllipseItem(-25, -25, 50, 50);
    b->target->setPen(QPen(Qt::red));
    b->onTouch = [this, bb=&*b] (QEvent::Type etype, QPointF p)
    {
      qDebug() << "catatouch " << etype << " " << p;
      auto it = std::find_if(buildings.begin(), buildings.end(), [&](auto const& c) { return c.get() == bb;});
      if (it == buildings.end())
      {
        qDebug() << "no hit";
        return;
      }
      auto b = *it;
      if (etype == QEvent::TouchEnd)
        return;
      if (b->handle != nullptr)
        delete b->handle;
      b->handle = game.scene.addEllipse(-25, -25, 50, 50);
      b->handle->setPos(p.x(), p.y());
      auto tt = std::make_shared<TouchTarget>();
      b->handleTouch = tt;
      auto sp = b->handle->scenePos();
      tt->px = sp.x();
      tt->py = sp.y();
      tt->radius = 25;
      tt->onTouch = [this, b](QEvent::Type t, QPointF p) { onHandle(b, t, p);};
      targets.push_back(tt);
    };
  }
  buildings.push_back(b);
}

void Player::onSymbol(int clsid, QPointF center)
{
  qDebug() << "Player::onSymbol " << clsid << " " << center;
  auto cost = C::symbolsManaCost[clsid];
  if (cost == -1)
    return;
  if (mana < cost)
  {
    helpText->setPlainText("Not enough mana!");
    return;
  }
  if (now() - symbolLastUsedTime[clsid] < std::chrono::milliseconds(C::symbolCooldownsMs[clsid]))
  {
    helpText->setPlainText("spell in cooldown!");
    return;
  }
  auto use = [&,this]() { mana -= cost; symbolLastUsedTime[clsid] = now();};
  Symbol s = (Symbol)clsid;
  if (s == Symbol::TriUp)
  {
    use();
    spawn(BuildingKind::Catapult, center);
  }
  else if (s == Symbol::Square)
  {
    if (farmsRemaining <= 0)
      helpText->setPlainText("No farm remaining");
    else
    {
      auto has = std::count_if(buildings.begin(), buildings.end(), [](auto const&b) {return b->kind == BuildingKind::Farm;});
      if (has >= C::farmsLiveMax)
        helpText->setPlainText("max number of live farms reached");
      else
      {
        use();
        spawn(BuildingKind::Farm, center);
        farmsRemaining--;
        farmsCounter->setPlainText(std::to_string(farmsRemaining).c_str());
      }
    }
  }
  else if (s == Symbol::Cross)
  {
    for (int i=0; i<buildings.size(); i++)
    {
      auto& b = *buildings[i];
      if (b.kind == BuildingKind::Farm)
        continue;
      if (norm2(center-QPointF(b.px, b.py)) < b.radius*b.radius)
      {
        delete b.pixmap;
        if (b.handle != nullptr)
          delete b.handle;
        if (b.target != nullptr)
          delete b.target;
        if (b.handleTouch)
          targets.swapOut(b.handleTouch);
        b.onTouch = nullptr;
        buildings.swapOut(i);
        use();
        break;
      }
    }
  }
  else if (s == Symbol::Peak)
  {
    auto affected = game.grid->aroundWide(center.x(), center.y(), C::waveRadius,
      [] (const ZombiePtr&) { return true;});
    for (auto& z: affected)
    {
      auto d = QPointF(z->px, z->py)-center;
      double f = (double)(rand()%1000)/1000.0*(C::waveDisplacmentFactorMax-C::waveDisplacmentFactorMin)+C::waveDisplacmentFactorMin;
      auto np = center + d*f;
      game.move(z, np);
    }
    use();
  }
  else if (s == Symbol::Plus)
  {
    for (auto& b: buildings)
    {
      if (QRectF(b->px-100, b->py-100, 200, 200).contains(center))
      {
        for (int i=0; i<C::burstZombieCount; i++)
          spawnZombie(b);
        use();
        break;
      }
    }
  }
  else if (s == Symbol::Waves)
  {
    for (auto& b: buildings)
    {
      if (QRectF(b->px-100, b->py-100, 200, 200).contains(center))
      {
        b->hiddenUntil = now() + std::chrono::milliseconds(C::hideDurationMs);
        use();
        break;
      }
    }
  }
}

void Player::spawnZombie(BuildingPtr where)
{
  auto z = std::make_shared<Zombie>(*this);
  z->px = where->px + ((double)(rand()%1000)/1000.0-0.5)*C::zombieSpawnSpread*2.0;
  z->py = where->py + ((double)(rand()%1000)/1000.0-0.5)*C::zombieSpawnSpread*2.0;
  z->hitpoints = C::zombieHitpoints;
  z->pixmap = game.scene.addEllipse(-5, -5, 10, 10, QPen(game.pColors[index]), QBrush(game.pColors[index]));
  z->pixmap->setPos(QPointF(z->px, z->py));
  z->pixmap->setZValue(10);
  zombies.push_back(z);
  game.grid->add(z);
}

void Player::tick()
{
  auto tme = now();
  double elapsed = (double)std::chrono::duration_cast<std::chrono::microseconds>(tme-lastTick).count()/1000000.0;
  lastTick = tme;

  // update mana
  mana = std::min(C::manaMax, mana + elapsed*C::manaRegenRate);
  manaRect->setRect(0,0, game.h * mana / C::manaMax, 50);
  // update cooldowns
  for (int i: C::activeSymbols)
  {
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(tme-symbolLastUsedTime[i]).count();
    if (elapsed > C::symbolCooldownsMs[i])
    {
      if (mana >= C::symbolsManaCost[i])
        cooldownRect[i]->setBrush(QBrush(QColor(0,255,0,25)));
      else
        cooldownRect[i]->setBrush(QBrush(QColor(255,0,255,60)));
    }
    else
    {
      long a = 255- elapsed * 230 / C::symbolCooldownsMs[i];
      a = std::min(255L, std::max(a, 0L));
      cooldownRect[i]->setBrush(QBrush(QColor(255,0,0,a)));
    }
  }
  // update smokes
  for (int i=0; i<smokes.size(); ++i)
  {
    auto elapsed = tme - smokes[i]->createdAt;
    if (elapsed >= std::chrono::milliseconds(C::smokeLifeTimeMs))
    {
      smokes.swapOut(i);
      i--;
      continue;
    }
    auto elapsedMS = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    double s = 1.0 - (double)elapsedMS/(double)C::smokeLifeTimeMs;
    smokes[i]->animation->setScale(s);
  }
  for (auto& b: buildings)
  {
    if (b->kind == BuildingKind::Farm && tme-b->lastSpawnTime > std::chrono::milliseconds(C::zombieSpawnIntervalMs))
    {
      spawnZombie(b);
      b->lastSpawnTime = tme;
    }
  }
}

struct Target
{
  unsigned int playerMask;
  double px, py;
  double weight;
};

void Game::move(ZombiePtr& z, QPointF pos)
{
  grid->move(z, pos.x(), pos.y());
  z->px = pos.x();
  z->py = pos.y();
  z->pixmap->setPos(pos);
}

void Game::tick()
{
  static Time globalStart = now();
  auto tme = now();
  double elapsed = (double)std::chrono::duration_cast<std::chrono::microseconds>(tme-lastTick).count()/1000000.0;
  double frameTime = (double)std::chrono::duration_cast<std::chrono::microseconds>(tme-globalStart).count()/1000000.0;
  lastTick = tme;
  if (tme - lastPlayerTick > std::chrono::milliseconds(100))
  {
    lastPlayerTick = tme;
    for (auto& p: players)
      p.tick();
  }
  // move zombies, precompute target list
  vector<Target> targets;
  for (auto&p : players)
  {
    for (auto& b: p.buildings)
    {
      if (b->hiddenUntil > tme || b->kind == BuildingKind::Catapult)
        continue;
      targets.push_back(Target{.playerMask = ~(1<<p.index), .px=b->px, .py=b->py, .weight = C::buildingWeight});
    }
    for (auto& s: p.smokes)
    {
      auto ltms = std::chrono::duration_cast<std::chrono::milliseconds>(tme-s->createdAt).count();
      auto w = C::smokeWeight * (1.0 - (double)ltms / (double)C::smokeLifeTimeMs);
      targets.push_back(Target{.playerMask = 1 << p.index, .px=s->px, .py=s->py, .weight = w});
    }
  }
  int zcount = 0, zmoved = 0;
  // move all zombies
  for (auto&p: players)
  {
    for (auto&z: p.zombies)
    {
      zcount++;
      Target* best = nullptr;
      double bestWeight = 0;
      for (auto& tgt: targets)
      {
        if ((tgt.playerMask & (1 << p.index)) == 0)
          continue;
        double w = tgt.weight / notZero(sqrt(norm2(QPointF(tgt.px, tgt.py)-QPointF(z->px, z->py))));
        if (w > bestWeight)
        {
          best = &tgt;
          bestWeight = w;
        }
      }
      // chek if there is a z nearby
      auto closest = grid->closestAround(z->px, z->py,  C::zombieAggroDistance,
        [&](ZombiePtr const& zz) { return zz->owner != p;});
      std::optional<QPointF> go;
      if (closest)
      {
        // you might be tempted to compute fight target now, in order to
        // "<<'optimize'>>" but this causes
        // desequilibrium that advantages player 2
        auto d2 = norm2(QPointF((*closest)->px, (*closest)->py)-QPointF(z->px, z->py));
        go = QPointF((*closest)->px, (*closest)->py);
      }
      else if (best)
        go = QPointF(best->px, best->py);
      if (go)
      {
        auto v = *go - QPointF(z->px, z->py);
        auto norm = sqrt(norm2(v));
        if (norm != 0)
        {
          zmoved++;
          v = v * C::zombieMoveSpeed * elapsed / norm;
          auto pos = QPointF(z->px, z->py) + v;
          move(z, pos);
        }
      }
    }
  }
  // fight
  for (auto&p: players)
  {
    for (auto&z: p.zombies)
    {
      auto closest = grid->closestAround(z->px, z->py,  C::zombieCombatDistance,
        [&](ZombiePtr const& zz) { return zz->owner != p;});
      if (!closest)
        continue;
      auto& target = *closest;
      // compute supports
      int zsupp = grid->countAround(z->px, z->py, C::zombieSupportDistance,
        [&](ZombiePtr const& b) { return b->owner == z->owner;});
      int tsupp = grid->countAround(target->px, target->py, C::zombieSupportDistance,
        [&](ZombiePtr const& b) { return b->owner == target->owner;});
      int dsup = zsupp-tsupp;
      double pOffset = (double)dsup/C::zombieOverwhelmingSupport;
      pOffset = std::max(-1.0, std::min(1.0, pOffset));
      double base = C::zombieBaseDmg + C::zombieExtraMaxDmg*pOffset/2.0;
      double bonus = C::zombieExtraMaxDmg;
      double roll = (double)(rand()% 1000)/1000.0 * bonus + base;
      target->hitpoints -= roll;
      qDebug() << frameTime << " " << z->owner.index << "/" << zsupp << " -> " << target->owner.index << "/" << tsupp << "  o " << pOffset << "  r " << roll << "  hp " << target->hitpoints;
    }
  }
  std::vector<QPointF> asplosions;
  // bring out your dead!
  for (auto&p: players)
  {
    for (int i=0; i<p.zombies.size(); ++i)
    {
      if (p.zombies[i]->hitpoints <= 0)
      {
        asplosions.push_back(QPointF(p.zombies[i]->px, p.zombies[i]->py));
        delete p.zombies[i]->pixmap;
        grid->remove(p.zombies[i]);
        p.zombies.swapOut(i);
        i--;
      }
    }
  }
  for (auto x: asplosions)
  {
    auto tgts = grid->around(x.x(), x.y(), C::zombieExplosionRadius, [](ZombiePtr const&){return true;});
    for (auto& z: tgts)
    {
      z->hitpoints -= C::zombieExplosionDamage;
      if (z->hitpoints <= 0)
      {
        delete z->pixmap;
        grid->remove(z);
        z->owner.zombies.swapOut(z);
      }
    }
  }
  // building damage
  for (auto& p: players)
  {
    for (int i=0; i<p.buildings.size();++i)
    {
      auto& b = p.buildings[i];
      auto zs = grid->countAround(b->px, b->py, b->radius,
        [&](ZombiePtr const& z) { return b->owner != z->owner;});
      if (zs)
      {
        b->hitpoints -= elapsed * (double)zs * C::zombieStructureDmg;
        b->hpBar->setRect(-100, 0, 200*b->hitpoints/C::buildingHitpoints[(int)b->kind], 25);
        if (b->hitpoints <= 0)
        { // FIXME DUP
          delete b->pixmap;
          if (b->handle != nullptr)
            delete b->handle;
          if (b->target != nullptr)
            delete b->target;
          if (b->handleTouch)
            p.targets.swapOut(b->handleTouch);
          b->onTouch = nullptr;
          p.buildings.swapOut(i);
          i--;
        }
      }
    }
  }
  //qDebug() << zcount << " " << zmoved << " " << std::chrono::duration_cast<std::chrono::microseconds>(now()-tme).count();
}
void Game::run(QApplication& app)
{
  for (int i=0; i<=8;i++)
    for (int p=0;p<4;p++)
      smoke[p].push_back(new QPixmap(("assets/smoka-" + std::to_string(i)+"-"+std::to_string(p)+".png").c_str()));
  for (int i=0; i < 15;i++)
    symbols.push_back(new QPixmap((std::string("assets/") + symbolNames[i] + ".png").c_str())); 
  w = 3840;
  h = 2160;
  grid = new Grid<ZombiePtr>(w, h, 216);
  float scale = 1;
  GraphicsView* view = new GraphicsView(*this, &scene);
  view->setSceneRect(0,0,w, h);
  scene.addPixmap(QPixmap("assets/bg1.jpg"));
  //view.scale(scale,scale);
  view->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  view->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  view->show();
  view->windowHandle()->setPosition(0, 0);
  view->resize(w*scale+1, h*scale+1);
  view->showFullScreen();
  players.reserve(2);
  players.emplace_back(0, *this, QRectF(0, 0, w/2, h), 90);
  players.emplace_back(1, *this, QRectF(w/2, 0, w, h), 270);
  auto timer = new QTimer();
  timer->connect(timer, &QTimer::timeout, std::bind(&Game::tick, this));
  timer->setInterval(20);
  timer->start();
  lastTick = now();
  lastPlayerTick = lastTick;
  app.exec();
}

int main(int argc, char **argv)
{
  QApplication app(argc, argv);
  Game* g = new Game();
  g->run(app);
}