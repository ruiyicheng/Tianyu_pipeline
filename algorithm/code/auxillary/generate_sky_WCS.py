import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from astropy.wcs import WCS
import pandas as pd
def fibonacci_covering(theta_rad):
    assert 0 < theta_rad < np.pi / 2, "Angular radius should be in (0, π/2) radians."
    n_points = int(1.2*np.ceil(4 / (theta_rad ** 2)))

    centers = []
    golden_angle = np.pi * (3 - np.sqrt(5))

    for i in range(n_points):
        z = 1 - 2 * i / float(n_points - 1)
        radius = np.sqrt(1 - z * z)
        theta = golden_angle * i
        x = np.cos(theta) * radius
        y = np.sin(theta) * radius
        centers.append((x, y, z))

    return centers

def plot_sphere_with_caps(centers):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1,1,1])

    # Plot unit sphere
    u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    ax.plot_surface(x, y, z, color='lightblue', alpha=0.2, linewidth=0)

    # Plot cap centers
    xs, ys, zs = zip(*centers)
    ax.scatter(xs, ys, zs, color='red', s=1, label='Cap Centers')

    ax.set_title("Spherical Cap Centers on Unit Sphere")
    ax.legend()
    plt.tight_layout()
    plt.show()

# Example usage
def generate_sky_WCS(NAXIS1,NAXIS2,pixel_scale,theta ,xflip=False):
   
    angular_radius_deg = pixel_scale/2*np.min([NAXIS1,NAXIS2])  # Convert to degrees
    theta_rad = np.radians(angular_radius_deg)
    centers = np.array(fibonacci_covering(theta_rad))
    #plot_sphere_with_caps(centers)
    print(f"Generated {len(centers)} cap centers.")
    
    x = centers[:, 0]
    y = centers[:, 1]
    z = centers[:, 2]

    ra = np.arctan2(y, x) * 180 / np.pi+180
    dec = np.arcsin(z) * 180 / np.pi
    radec = np.hstack((ra.reshape(-1,1), dec.reshape(-1,1)))
    # Have some redundant info if it is for the same instrumentation
    WCS_info_dict = {"sky_id":[],"target_name":[],"ra":[],"dec":[]}
    i = 2
    for ra_this,dec_this in radec:
        WCS_info_dict['sky_id'].append(i)
        WCS_info_dict['target_name'].append(f"sky_{ra_this}_{dec_this}")
        WCS_info_dict['ra'].append(ra_this)
        WCS_info_dict['dec'].append(dec_this)
        i+=1
    WCS_info_dict = pd.DataFrame(WCS_info_dict)
    return WCS_info_dict
        



# NAXIS1 = 9576 # Number of pixel of our telescope
# NAXIS2 = 6388 # Number of pixel of our telescope
# pixel_scale = 3.76 / 2563000 * 180 / np.pi # deg per pixel
NAXIS1 = 8120 # Number of pixel of our telescope
NAXIS2 = 8120 # Number of pixel of our telescope
pixel_scale = 10 / 1560000 * 180 / np.pi # deg per pixel
theta = 0
pd_WCS = generate_sky_WCS(NAXIS1,NAXIS2,pixel_scale,theta)
origin = pd.read_csv('/home/test/workspace/Tianyu_pipeline/algorithm/data/testinput/metadata/sky/demo_sky.csv')
pd_WCS = pd.concat([origin,pd_WCS],axis=0,ignore_index=True)
pd_WCS.to_csv('/home/test/workspace/Tianyu_pipeline/algorithm/data/testinput/metadata/sky/demo_sky.csv', index=False)
# # plot the WCS regions
# def plot_WCS_region(row_of_wcs):
#     wcs = WCS(naxis=2)
#     wcs.wcs.crpix = [row_of_wcs['CRPIX1'], row_of_wcs['CRPIX2']]
#     wcs.wcs.cd = [[row_of_wcs['CD1_1'], row_of_wcs['CD1_2']],[row_of_wcs['CD2_1'], row_of_wcs['CD2_2']]]
#     wcs.wcs.crval = [row_of_wcs['CRVAL1'], row_of_wcs['CRVAL2']]
#     wcs.wcs.ctype = [row_of_wcs['CTYPE1'], row_of_wcs['CTYPE2']]
#     NAXIS1 = row_of_wcs["NAXIS1"]
#     NAXIS2 = row_of_wcs["NAXIS2"]
#     pixels = np.array([[0, 0], [NAXIS1, 0], [NAXIS1, NAXIS2], [0, NAXIS2],[0, 0]])
    
#     res = wcs.all_pix2world(pixels,0)/180*np.pi
#     res[:,0] = res[:,0] - np.pi

#     print(res)
#     plt.plot(np.squeeze(res[:,0]),np.squeeze(res[:,1]),linestyle='-',alpha=0.5)
# plt.figure()
# ax = plt.subplot(111, projection="aitoff")
# #ax.set_xticklabels(['14h','16h','18h','20h','22h','0h','2h','4h','6h','8h','10h'])
# plt.xlabel(r"$\alpha$")
# plt.ylabel(r'$\delta$')
# ct = 0
# for i,r in pd_WCS.iterrows():
#     ct +=1
#     plot_WCS_region(r)
#     # if ct>20:
#     #     break
# plt.show()