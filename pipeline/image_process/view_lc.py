import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('/dssg/home/acct-tdlffb/tdlffb-user1/workspace/TianYu/image_process/out/light_curve/mgo/sky1/20231023-20231024.csv')
target = data[data['target']=='MG-S1-59']
flux = target['flux_SE']
time = target['JD_start']+target['JD_end']
plt.plot(time,flux,'.')
plt.xlabel('JD')
plt.ylabel('SE_flux')
plt.title('light curve of MG-S1-59')
plt.savefig('demo.pdf')
