#gebruik dit fault-proof ding om de niet-fault-proof-echte-versie aan te passen
def make_plot(tit):
		x = np.arange(10)

		fig = plt.figure()
		ax = plt.subplot(111)

		one, = ax.plot(x, 1 * x, label='$y = 1x$')
		two, = ax.plot(x, 2 * x, label='$y = 2x$')

		legend1 = ax.legend(handles = [one,two], loc='center left', bbox_to_anchor=(1, 0.5))
		axx = plt.gca().add_artist(legend1)

		xa = ax.twinx()

		five, = xa.plot(x, 5 * x, label='$y = 5x$')
		six, = xa.plot(x, 6 * x, label='$y = 6x$')


		legend2 = plt.legend(handles=[five,six],loc = 'center right', bbox_to_anchor=(1,0.5))

		box = ax.get_position()
		ax.set_position([box.x0, box.y0, box.width, box.height])
		box = xa.get_position()
		xa.set_position([box.x0, box.y0, box.width, box.height])
		
		plt.title(tit)
		plt.show()
		#pp.savefig(fig, dpi = 300, transparent = True)



from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
		
plt.close()
#pp = PdfPages("this-is-a-test.pdf")
titlist = ['this', 'is', 'a', 'test']
for tit in titlist:
	make_plot(tit)
#pp.close()