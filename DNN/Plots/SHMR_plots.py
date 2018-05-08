import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



TRUE_data = np.genfromtxt('true_halo_mass_test_set46.csv',
                            delimiter=',',
                            names={'True_Values', 'Original_Index'})

PREDICT_data = np.genfromtxt('NN_halo_mass_test_set46.csv',
                                delimiter=',',
                                names=True)

FEATURES_data = np.genfromtxt('features_test_set46.csv',
                                delimiter=',',
                                names=True)


plt.figure(1,(14,10))

plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.tick_params(which='minor',width=1,length=3)
plt.tick_params(which='major',width=1,length=7)



plt.xscale('log')
plt.xlim(10**-2.1, 10**4.7)
plt.xlabel('Halo Mass ($10^{10}$ $M_{\odot}$/$h$)', fontsize=20)

plt.yscale('log')
plt.ylim(10**-3.2, 10**3)
plt.ylabel('Stellar Mass ($10^{10}$ $M_{\odot}$/$h$)', fontsize=20)



plt.scatter(TRUE_data['True_Values'], FEATURES_data['SubhaloStellarPhotometricsMassInRad'],
                s=50,
                color='#276ab3',        # or 5b7c99
                label='Illustris Simulation Data',
                #marker=',',
                alpha=0.6)

plt.scatter(PREDICT_data['predictions'], FEATURES_data['SubhaloStellarPhotometricsMassInRad'],
                s=50,
                color='#fcb001',     # or fcc006,
                label='Neural Network Predictions',
                #marker=',',
                alpha=0.8)



plt.legend(loc=1, prop={'size': 20})



plt.text(0.3,150.0,'Input Features:', horizontalalignment='right', fontsize=20)
plt.text(0.03,85.0,'1-8.  Halo Stellar Photometrics (8 bands) ', horizontalalignment='left', fontsize=15)
plt.text(0.03,55.0,'9.  Halo Star Formation Rate', horizontalalignment='left', fontsize=15)
plt.text(0.03,35.0,'10.  Halo Gas Metallicity', horizontalalignment='left', fontsize=15)
plt.text(0.03,22.0,'11.  Halo "Apparent Size"', horizontalalignment='left', fontsize=15)
plt.text(0.03,14.0,'12.  Halo Stellar Mass', horizontalalignment='left', fontsize=15)




plt.show()
