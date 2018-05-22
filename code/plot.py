import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['savefig.dpi'] = 500
plt.style.use('ggplot')
plt.rcParams["errorbar.capsize"] = 3


'''
Example output file:
{
	"d_loss": [0.4429759755730629, 0.5224442854523659, 0.6404945626854897, 0.600304365158081, 0.5165300965309143],
	"g_loss": [0.4260864853858948, 0.3554322421550751, 0.36993931606411934, 0.30190570279955864, 0.36014848947525024],
	"is": [
		[1.028869390487671, 0.03082055225968361],
		[1.1018842458724976, 0.07953199744224548],
		[1.0921385288238525, 0.0702158510684967],
		[1.0412414073944092, 0.03792886435985565],
		[1.02135169506073, 0.01158658042550087]
	]
}
'''

DO_INCEPTION_SCORE = True
DO_LOSSES = True
FIGSIZE = (10,4)

def concat():
    filenames = ['w-wc-dcgan_cifar10_0', 'w-wc-dcgan_cifar10_1', 'w-wc-dcgan_cifar10_2', 'w-wc-dcgan_cifar10_3']
    all_data = {
        'd_loss': [],
        'g_loss': [],
        'is': []
    }
    for filename in filenames:
        data = {}
        with open('results/' + filename) as f:
            data = json.load(f)
        all_data['d_loss'] = all_data['d_loss'] + data['d_loss']
        all_data['g_loss'] = all_data['g_loss'] + data['g_loss']
        all_data['is'] = all_data['is'] + data['is']


        with open('results/w-wc-dcgan_cifar10', 'w') as f:
            json.dump(all_data, f)

def plot_single():
    filenames = ['w-dcgan_cifar10']#['vanilla_gan'] # sn_gan

    for filename in filenames:
        
        data = {}
        with open('results/' + filename) as f:
            data = json.load(f)

        # Inception score
        if DO_INCEPTION_SCORE:
            series = []
            series_err = []
            for score in data['is']:
                series.append(score[0])
                series_err.append(score[1])

            x = np.arange(len(series))
            plt.figure(figsize=FIGSIZE)
            plt.xlabel('Epoch')
            plt.ylabel('Inception score')
            plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.15)
            plt.errorbar(x, series, yerr=series_err, elinewidth=1)
            plt.savefig('results/figures/' + filename + '_is.png')
            # plt.show()

        if DO_LOSSES:
            # Losses
            d_losses = []
            g_losses = []
            x = np.arange(len(data['d_loss']))
            for i in x:
                d_losses.append(data['d_loss'][i])
                g_losses.append(data['g_loss'][i])

            fig, ax1 = plt.subplots(1,1,figsize=FIGSIZE)
            ax1.set_xlabel('Epoch')
            ax1.plot(x, d_losses, '-')
            ax1.set_ylabel('Discriminator loss', color='#f8766d')

            ax2 = ax1.twinx()
            ax2.plot(x, g_losses, 'b-')
            ax2.set_ylabel('Generator loss', color='b')

            fig.tight_layout()
            plt.savefig('results/figures/' + filename + '_losses.png')
            # plt.show()
    
def plot_multiple():
    filenames = ['sn_gan_concat', 'vanilla_gan']

    
    plt.figure(figsize=FIGSIZE)
    plt.xlabel('Epoch')
    plt.ylabel('Inception score')
    plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.15)
    for filename in filenames:
        
        data = {}
        with open('results/' + filename) as f:
            data = json.load(f)

        # Inception score
        if DO_INCEPTION_SCORE:
            series = []
            series_err = []
            for score in data['is']:
                series.append(score[0])
                series_err.append(score[1])

            x = np.arange(len(series))
            #plt.errorbar(x, series, yerr=series_err, elinewidth=1)
            plt.plot(x, series)#, yerr=series_err, elinewidth=1)
    plt.legend(['SNDCGAN', 'DCGAN'])
    plt.savefig('results/figures/multiple.png')
            # plt.show()


if __name__ == '__main__':  

    #concat()
    # plot_multiple()
    plot_single()
    