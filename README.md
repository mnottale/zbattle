# ZBattle

ZBattle is a two players game for big touchscreen in which you spawn zombies
and cast spells in order to overwhelm your adversary and destroy his/her zombie
farms.

![Screenshot](medias/zbattle.png?raw=true "Screenshot")

## Warning

The game is in early stage and not yet balanced at all.

## Requirements

Qt, Python3, pytorch, torchvision, numpy, PIL.

## Compilation

    qmake zbattle.pro
    make

## Running

First run `serve.py models/model-100`, then ./zbattle. The first command spawn a
server that runs a Deep Neural Network that will recognize the shapes you draw.
How cool is that!

zbattle will expect screen at 0,0 coordinates to be a 4K monitor, edit run() to
change the resolution and screen coordinates if needed.

## Playing

The red bar at the end of the screen is your mana bar. The shapes are the things
you can do by drawing them at any scale on your half of the screen.
Each drawing costs mana, and has a specific cooldown.

The symbols will turn purple if you do not have enough mana, or red if in cooldown.

### Square: zombie farm

Spawn a zombie farm. You can have only a fixed amount of them present at the same time,
and also have a total number of build limits. You lose when all your farms are destroyed.

Farms spawn zombies at regular interval.

Zombies are quite dumb and will walk toward the closest ennemy farm, or ennemy
zombie if close enough.

Be careful: zombies explode when dying, dealing a small amount of damages to all
zombies in a short radius. Avoid too large groups!

### Triangle pointing up: smoke bomb catapult

You can use the catapult to throw smoke bombs that will attract your zombies
toward them. Touch the catapult and drag opposite to where you want to shoot.
A red circle will show where the smoke bomb would land. To cancel drag back to
the center of the catapult and release.

Buttons on your end of the screen give you some control on the smoke effect:

- PER/TRA: persistant or transient smoke. Zs will stay on a persistent smoke,
but will leave a transient one as soon as they reach the center, and then ignore
that specific smoke forever
- D+/D-: change the life time of the smoke
- P+/P-: change the attractive power of a smoke. For example a value of two means
Zs will prefer smoke over ennemy farm if the smoke is less than twice as far.

### Small Hill: shockwave

Sends a shockwave that repulses all zombies by a small distance. Useful to gain
some time, or spread zombies that are grouped together.

### Waves: stealth

Makes your building invisible to ennemy zombies for a short period of time.

### Plus: burst spawn

Spawn a burst of zombies on one of your buildings.

### X Cross: delete a building excluding farms

Here just in case you need it.

