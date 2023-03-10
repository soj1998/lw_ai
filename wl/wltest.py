import logging
import wltimer
import syngedata as syn
import mymatplotlib as plt

logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


t = wltimer.Timer()
t.start()
g = syn.YiYuanSj(w=2, b=3)
x = g.x
y = g.y
logging.info(len(x))
logging.info(len(y))
pl = plt.MyMatplotlib()
pl.scatterplot(x, y)


haoshi = t.stop()
logging.info(haoshi)


