import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

red_patch = mpatches.Patch(color='red', label='NN Prediction Data')
green_patch = mpatches.Patch(color='green', label='True Illustris Data')



TRUE_data = np.genfromtxt('true_halo_mass_test_set18.csv',
                            delimiter=',',
                            names={'True_Values', 'Original_Index'})

PREDICT_data = np.genfromtxt('NN_halo_mass_test_set18.csv',
                                delimiter=',',
                                names=True)

FEATURES_data = np.genfromtxt('features_test_set18.csv',
                                delimiter=',',
                                names=True)


plt.title('Stellar Mass vs Halo Mass (Predicted and True Illustris)')
plt.legend(handles=[red_patch, green_patch])

plt.xscale('log')
plt.xlim(10**-2, 10**5)
plt.xlabel('Halo Mass (10^10 Mstar/h)')

plt.yscale('log')
plt.ylabel('Stellar Mass (10^10 Mstar/h)')

plt.scatter(TRUE_data['True_Values'], FEATURES_data['SubhaloStellarPhotometricsMassInRad'], color='green')
plt.scatter(PREDICT_data['predictions'], FEATURES_data['SubhaloStellarPhotometricsMassInRad'], color='red')


plt.show()
