import os
import sys
sys.path.append(os.path.abspath("/home/test/workspace/Tianyu_pipeline/algorithm/code"))
import sep
import astropy.io.fits as fits
from astropy.wcs import WCS
import utils.dataloader as dataloader
import utils.Bertin as Bertin
import numpy as np
import pandas as pd
import pyfftw.interfaces.numpy_fft as fft
import pyfftw
from scipy.ndimage import convolve
import middleware.pipeline_component as pipeline_component
from scipy.ndimage import label, center_of_mass,generate_binary_structure
import cv2
import matplotlib.pyplot as plt
class differential_image(pipeline_component.pipeline_component):
    def __init__(self,free = True, n_divide = 5):
        super().__init__(free = free)
        self.n_divide = n_divide
    
    def get_boundary(self,x_n,y_n,x_shape,y_shape):
        x_width = x_shape // self.n_divide + 1
        y_width = y_shape // self.n_divide + 1
        x_start = x_n * x_width
        y_start = y_n * y_width
        x_end = (1+x_n) * x_width
        y_end = (1+y_n) * y_width
        return x_start, y_start, x_end, y_end

    def set_template(self,template_path,sources):
        weight_path = template_path.replace(".fit", "_mask.fits")
        self.template_path = template_path
        template_weight = fits.getdata(weight_path)#.astype(bool)
        self.sources = sources
        self.template_image = fits.getdata(template_path)
        self.template_header = fits.getheader(template_path)
        #self.template_wcs = dataloader.dataloader.get_wcs_from_header(self.template_header)
        self.template_image = self.template_image.byteswap().newbyteorder()
        mean_weight = np.mean(template_weight)
        std_weight = np.std(template_weight)
        mask_bkg = template_weight<mean_weight-0.5*std_weight
        plt.figure()
        plt.imshow(mask_bkg, origin='lower', cmap='gray',vmin = 0 , vmax = 1)
        plt.title('Mask Background')
        plt.colorbar()
        plt.savefig(os.path.join(self.data_loader.output_plot_path,'mask_bkg.pdf'))
        self.template_bkg = sep.Background(self.template_image, mask = mask_bkg)

        self.template_image_debackgrounded = self.template_image - self.template_bkg.back()
        self.data_new_bkg_std = self.template_bkg.rms()
        self.template_local_metadata = self.estimate_parameters(self.template_path,self.template_image_debackgrounded, self.template_bkg.rms(), self.template_header)
        # template_local_metadata goes:
        # {(x_divide,y_divide,attribute):value}
    def ZOGY_algo (self,data_ref, data_new, psf_ref, psf_new, data_ref_bkg_std,
                data_new_bkg_std, fr, fn, use_FFTW=False, nthreads=1,nx = 0,ny = 0,plot=False):

        """function to run ZOGY on a subimage"""

        # option 1: set f_ref to unity
        #f_ref = 1.
        #f_new = f_ref * np.mean(fratio_sub)
        # option 2: set f_new to unity
        # fn = 1.
        # fr = fn / fratio
        #dx = dx
        #dy = dy

        N = np.copy(data_new)
        R = np.copy(data_ref)
        Pn = psf_new
        Pr = psf_ref

        # before running zogy, pixels with zero values in ref need to
        # be set to zero in new as well, and vice versa, to avoid
        # subtracting non-overlapping image part
        mask_zero = ((R==0) | (N==0))
        N[mask_zero] = 0
        R[mask_zero] = 0

        # determine subimage s_new and s_ref from background RMS
        # images
        if np.sum(~mask_zero) != 0:

            sn = np.median(data_new_bkg_std[~mask_zero])
            sr = np.median(data_ref_bkg_std[~mask_zero])
            # try providing full subimages for sn and sr
            #sn = data_new_bkg_std
            #sr = data_ref_bkg_std

        else:

            sn = 1
            sr = 1


        # variance estimate: background-subtracted image +
        # measured background variance
        Vn = N + data_new_bkg_std**2
        Vr = R + data_ref_bkg_std**2


        # boolean [use_FFTW] determines if initial forward fft2 is
        # initialized using pyfftw.FFTW or not; due to planning involved
        # this speeds up all subsequent calls to convenience function
        # [fft.fft2] significantly, with a loop time of 0.2s instead of
        # 0.3s.  If nthreads>1 then this speed-up becomes less dramatic,
        # e.g. with 4 threads, the loop time is 0.2s without [use_FFTW]
        # and 0.17s with [use_FFTW]. Sometimes, this seems to change the
        # results for the 1st subimage, in the same way for planner flags
        # FFTW_ESTIMATE (timing: ~97s on macbook) and FFTW_ESTIMATE
        # (~90s), but much worse for FFTW_PATIENT (~360s). So if this is
        # switched on, need to do a pre-processing of the 1st subimage.
        if use_FFTW:
            R = R.astype('complex64')
            R_hat = np.zeros_like(R)
            fft_forward = pyfftw.FFTW(R, R_hat, axes=(0,1), direction='FFTW_FORWARD',
                                    flags=('FFTW_ESTIMATE', ),
                                    threads=nthreads, planning_timelimit=None)
            fft_forward()
        else:
            R_hat = fft.fft2(R, threads=nthreads)

        N_hat = fft.fft2(N, threads=nthreads)

        Pn_hat = fft.fft2(Pn, threads=nthreads)
        #if get_par(set_zogy.psf_clean_factor,tel)!=0:
        #Pn_hat[Pn_hat<0] = 1e-6
        Pn_hat2_abs = np.abs(Pn_hat**2)

        Pr_hat = fft.fft2(Pr, threads=nthreads)
        #if get_par(set_zogy.psf_clean_factor,tel)!=0:
        #Pr_hat[Pr_hat<0] = 1e-6
        Pr_hat2_abs = np.abs(Pr_hat**2)

        sn2 = sn**2
        sr2 = sr**2
        fn2 = fn**2
        fr2 = fr**2
        fD = (fr*fn) / np.sqrt(sn2*fr2+sr2*fn2)

        denominator = (sn2*fr2)*Pr_hat2_abs + (sr2*fn2)*Pn_hat2_abs

        D_hat = (fr*(Pr_hat*N_hat) - fn*(Pn_hat*R_hat)) / np.sqrt(denominator)

        if use_FFTW:
            D = np.zeros_like(D_hat)
            fft_backward = pyfftw.FFTW(D_hat, D, axes=(0,1), direction='FFTW_BACKWARD',
                                    flags=('FFTW_MEASURE', ),
                                    threads=nthreads, planning_timelimit=None)
            fft_backward()
            D = np.real(D) / fD
        else:
            D = np.real(fft.ifft2(D_hat, threads=nthreads)) / fD

        P_D_hat = (fr*fn/fD) * (Pr_hat*Pn_hat) / np.sqrt(denominator)
        #P_D = np.real(fft.ifft2(P_D_hat, threads=nthreads))

        S_hat = fD*D_hat*np.conj(P_D_hat)
        S = np.real(fft.ifft2(S_hat, threads=nthreads))

        # alternative way to calculate S
        #S_hat = (fn*fr2*Pr_hat2_abs*np.conj(Pn_hat)*N_hat -
        #         fr*fn2*Pn_hat2_abs*np.conj(Pr_hat)*R_hat) / denominator
        #S = np.real(fft.ifft2(S_hat), threads=nthreads)

        # PMV 2017/01/18: added following part based on Eqs. 25-31
        # from Barak's paper
        kr_hat = (fr*fn2)*np.conj(Pr_hat)*Pn_hat2_abs / denominator
        kr = np.real(fft.ifft2(kr_hat, threads=nthreads))
        kr2 = kr**2
        kr2_hat = fft.fft2(kr2, threads=nthreads)

        kn_hat = (fn*fr2)*np.conj(Pn_hat)*Pr_hat2_abs / denominator
        kn = np.real(fft.ifft2(kn_hat, threads=nthreads))
        kn2 = kn**2
        kn2_hat = fft.fft2(kn2, threads=nthreads)

        Vr_hat = fft.fft2(Vr, threads=nthreads)
        Vn_hat = fft.fft2(Vn, threads=nthreads)

        VSr = np.real(fft.ifft2(Vr_hat*kr2_hat, threads=nthreads))
        VSn = np.real(fft.ifft2(Vn_hat*kn2_hat, threads=nthreads))

        # dx2 = dx**2
        # dy2 = dy**2
        # and calculate astrometric variance
        # Sn = np.real(fft.ifft2(kn_hat*N_hat, threads=nthreads))
        # dSndy = Sn - np.roll(Sn,1,axis=0)
        # dSndx = Sn - np.roll(Sn,1,axis=1)
        # VSn_ast = dx2 * dSndx**2 + dy2 * dSndy**2

        # Sr = np.real(fft.ifft2(kr_hat*R_hat, threads=nthreads))
        # dSrdy = Sr - np.roll(Sr,1,axis=0)
        # dSrdx = Sr - np.roll(Sr,1,axis=1)
        # VSr_ast = dx2 * dSrdx**2 + dy2 * dSrdy**2



        # and finally Scorr
        V_S = VSr + VSn
        #V_ast = VSr_ast + VSn_ast
        V = V_S #+ V_ast
        #Scorr = S / np.sqrt(V)
        # make sure there's no division by zero
        Scorr = np.copy(S)
        #Scorr[V>0] /= np.sqrt(V[V>0])
        mask = (V>0)
        Scorr[mask] /= np.sqrt(V[mask])

        # PMV 2017/03/05: added following PSF photometry part based on
        # Eqs. 41-43 from Barak's paper
        F_S = fn2*fr2*np.sum((Pn_hat2_abs*Pr_hat2_abs) / denominator)
        # divide by the number of pixels in the images (related to do
        # the normalization of the ffts performed)
        F_S /= R.size
        # an alternative (slower) way to calculate the same F_S:
        #F_S_array = fft.ifft2((fn2*Pn_hat2_abs*fr2*Pr_hat2_abs) / denominator)
        #F_S = F_S_array[0,0]

        alpha = S / F_S
        alpha_std = np.zeros(alpha.shape)
        #alpha_std[V_S>=0] = np.sqrt(V_S[V_S>=0]) / F_S
        mask = (V_S>=0)
        alpha_std[mask] = np.sqrt(V_S[mask]) / F_S

        if plot:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(15, 10))
            plt.subplot(2, 3, 1)
            mr = np.mean(R)
            nr = np.std(R)
            plt.imshow(R, origin='lower', cmap='gray', vmin=mr-nr, vmax=mr+nr)
            plt.title('Reference Image')
            plt.colorbar()

            plt.subplot(2, 3, 2)
            mn = np.mean(N)
            nn = np.std(N)
            plt.imshow(N, origin='lower', cmap='gray', vmin=mn-nn, vmax=mn+nn)
            plt.title('New Image')
            plt.colorbar()

            plt.subplot(2, 3, 3)
            md = np.mean(D)
            sd = np.std(D)
            plt.imshow(D, origin='lower', cmap='gray',vmin=md-sd, vmax=md+sd)
            plt.title('Difference Image D')
            plt.colorbar()

            plt.subplot(2, 3, 4)
            ms = np.mean(S)
            ss = np.std(S)
            plt.imshow(S, origin='lower', cmap='gray', vmin=ms-ss, vmax=ms+ss)
            plt.title('Source Image S')
            plt.colorbar()

            plt.subplot(2, 3, 5)
            mc = np.mean(Scorr)
            sc = np.std(Scorr)
            # s = np.ones((5,5))
            
            feature = Scorr>5
            # # morphology close
            # feature = cv2.morphologyEx(feature.astype(np.uint8), cv2.MORPH_CLOSE, s)
            s = np.ones((3,3))
            labeled_array, num_features = label(feature,structure=s)
            print(f'Number of features in Scorr: {num_features}')
            plt.imshow(Scorr>5, origin='lower', cmap='gray', vmin=0, vmax=1)
            plt.title('Corrected Source Image Scorr')
            plt.colorbar()

            plt.subplot(2, 3, 6)
            ma = np.mean(alpha)
            sa = np.std(alpha)
            plt.imshow(alpha, origin='lower', cmap='gray', vmin=ma-sa, vmax=ma+sa)
            plt.title('Alpha Image')
            plt.colorbar()

            plt.tight_layout()
            print('writing zogy output to zogy_output.pdf')
            savepath = os.path.join(self.data_loader.output_plot_path,f'zogy_output_{nx}_{ny}.pdf')
            plt.savefig(savepath)
            print('done')
        #return D, S, Scorr, alpha, alpha_std
        return D, Scorr, alpha, alpha_std

    def estimate_parameters(self,img_path,image_debkged,image_bkgrms, header):
        # input: resampled image, header, mask, sources
        # return {(x,y,attribute):value} attributes include: data, PSF, bkgstd, F
        # step 1: estimate PSF using PSFex
        def pad_psf_for_fft(psf_small, target_shape):
            psf_h, psf_w = psf_small.shape
            img_h, img_w = target_shape
            if psf_h > img_h or psf_w > img_w:
                raise ValueError("PSF dimensions must be smaller than or equal to target shape.")
            psf_padded = np.zeros(target_shape, dtype=psf_small.dtype)
            psf_padded[0:psf_h, 0:psf_w] = psf_small
            center_y = psf_h // 2
            center_x = psf_w // 2
            psf_padded = np.roll(psf_padded, -center_y, axis=0)
            psf_padded = np.roll(psf_padded, -center_x, axis=1)
            return psf_padded
        
        psf_path,psf_function = self.Bertin.PSFex_image(img_path,{},{})
        # remove psf_path
        os.remove(psf_path)
        wcs = WCS(header)
        x_star, y_star = wcs.all_world2pix(self.sources['ra'], self.sources['dec'], 0)
        sources = self.sources.copy()
        sources['x'] = x_star
        sources['y'] = y_star
        metadata = {}
        for x_n in range(self.n_divide):
            for y_n in range(self.n_divide):
                x_start, y_start, x_end, y_end = self.get_boundary(x_n,y_n,image_debkged.shape[1],image_debkged.shape[0])
                x_mid = (x_start + x_end) // 2
                y_mid = (y_start + y_end) // 2
                source_this_region = sources[(sources['x'] >= x_start) & (sources['x'] < x_end) & (sources['y'] >= y_start) & (sources['y'] < y_end)]
                if len(source_this_region) == 0:
                    print('Not enough sources in this region, skipping')
                    continue
                f,_,_ = sep.sum_circle(image_debkged, source_this_region['x'], source_this_region['y'], 15, err=True)
                F = np.mean(f)
                print(f'F_{x_n}_{y_n} = {F}')
                PSF = psf_function(x_mid, y_mid)
                PSF = PSF / np.sum(PSF)  # normalize the PSF
                image_debkg_this = image_debkged[y_start:y_end,x_start:x_end]
                PSF_padded = pad_psf_for_fft(PSF,image_debkg_this.shape)
                # step 5: save the parameters
                metadata[(x_n,y_n,'data')] = image_debkg_this
                metadata[(x_n,y_n,'PSF')] = PSF_padded
                metadata[(x_n,y_n,'bkgstd')] = image_bkgrms[y_start:y_end,x_start:x_end]
                metadata[(x_n,y_n,'F')] = F
        #print(metadata)
        return metadata
    
    def resample_image(self,new_image_path,template_path):
        # input: resampled image, header, mask, sources
        # return PSF, F, bkgstd
        def interpolate_2d(array,mask):
            kernel_size = 5
            kernel = np.ones((kernel_size,kernel_size),dtype=float)
            kernel[kernel_size//2, kernel_size//2] = 0  # center pixel is not counted
            
            # Replace NaNs with 0 for the purpose of summing
            mask_this = ~mask
            mask = mask.astype(float)
            # array_zeroed = np.where(mask_this, 0, array)
            # mask_nan = np.isnan(array)
            array_zeroed = np.where(mask_this, 0, array)

            # Convolve the array and the mask to count valid neighbors
            neighbor_sum = convolve(array_zeroed, kernel, mode='constant', cval=0.0)
            valid_neighbors = convolve(mask, kernel, mode='constant', cval=0.0)

            # Avoid division by zero

            interpolated_values = neighbor_sum / valid_neighbors


            filled_array = np.where(mask_this & (valid_neighbors > 0), interpolated_values, array_zeroed)
            
            # Fill NaNs with 0
            mask_nan = np.isnan(filled_array)
            filled_array = np.where(mask_nan, 0, filled_array)
            return filled_array
        mask_path = new_image_path.replace(".fit", "_mask.fits")
        mask = fits.getdata(mask_path).astype(bool)
        calibrated_image = fits.getdata(new_image_path)
        calibrated_image = calibrated_image.byteswap().newbyteorder()
        # remove the bkg of the calibrated image
        # calibrated_image_bkg = sep.Background(calibrated_image)
        # calibrated_image = calibrated_image - calibrated_image_bkg.back()
        calibrated_image_header = fits.getheader(new_image_path)

        interpolated_img = interpolate_2d(calibrated_image,mask)
        temp_interpolated_img_path = os.path.join(self.data_loader.output_cache_path,f"temp_calibrated_{os.getpid()}.fits")
        # save the interpolated image to the cache path
        fits.writeto(temp_interpolated_img_path, interpolated_img, header=calibrated_image_header, overwrite=True)
        # resample to template image using Bertin SWarp
        print([template_path,temp_interpolated_img_path])
        header_template = fits.getheader(template_path)
        #resampled_image_path,resampled_image_path_mask = self.Bertin.SWARP_stack([template_path,temp_interpolated_img_path], {},delete_weight = True,weight_number_list = [0,1],remove_cache_header = True)
        resampled_image_path,resampled_image_path_mask = self.Bertin.SWARP_stack([temp_interpolated_img_path], {},delete_weight = True,weight_number_list = [1],remove_cache_header = True,header_out = header_template)

        os.remove(resampled_image_path_mask)
        os.remove(temp_interpolated_img_path)
        return resampled_image_path
    def differential_image_algo(self,new_image_path):
        resampled_image_path = self.resample_image(new_image_path,self.template_path)
        new_image = fits.getdata(resampled_image_path)
        new_header = fits.getheader(resampled_image_path)
        # rmove the temporary resampled image
        
        new_image = new_image.byteswap().newbyteorder()
        mask_bkg = (new_image>-1e-5) & (new_image<1e-5)
        new_bkg = sep.Background(new_image,mask = mask_bkg)
        new_image_debackgrounded = new_image - new_bkg.back()
        # show bkg
        # bkg = new_bkg.back()
        # m = np.mean(bkg)
        # s = np.std(bkg)
        # plt.figure()
        # plt.imshow(bkg, origin='lower', cmap='gray', vmin=m-s, vmax=m+s)
        # plt.title('Background Image')
        # plt.colorbar()
        # plt.savefig(f'bkg_image_{os.getpid()}.pdf')
        # plt.figure()
        # plt.imshow(mask_bkg, origin='lower', cmap='gray', vmin=0, vmax=1)
        # plt.title('mask Image')
        # plt.colorbar()
        # plt.savefig(f'mask_image_{os.getpid()}.pdf')
        new_bkg_std = new_bkg.rms()
        new_local_metadata = self.estimate_parameters(resampled_image_path,new_image_debackgrounded, new_bkg_std, new_header)
        os.remove(resampled_image_path)
        ret_dict = {}
        item = ['data', 'PSF', 'bkgstd', 'F']
        for nx in range(self.n_divide):
            for ny in range(self.n_divide):
                # to check if the template metadata is available

                # run ZOGY algorithm
                for item_name in item:
                    if not (nx, ny, item_name) in self.template_local_metadata or not (nx, ny, item_name) in new_local_metadata:
                        print(f"Missing template metadata for ({nx}, {ny}, {item_name}), skipping")
                        continue
                D, Scorr, alpha, alpha_std = self.ZOGY_algo(
                    self.template_local_metadata[(nx, ny, 'data')],
                    new_local_metadata[(nx, ny, 'data')],
                    self.template_local_metadata[(nx, ny, 'PSF')],
                    new_local_metadata[(nx, ny, 'PSF')],
                    self.template_local_metadata[(nx, ny, 'bkgstd')],
                    new_local_metadata[(nx, ny, 'bkgstd')],
                    self.template_local_metadata[(nx, ny, 'F')],
                    new_local_metadata[(nx, ny, 'F')],
                    use_FFTW=False,
                    nthreads=1,
                    plot=True,
                    nx=nx,
                    ny=ny
                )
                # save the results
                ret_dict[(nx,ny,'D')] = D
                ret_dict[(nx,ny,'Scorr')] = Scorr
                ret_dict[(nx,ny,'alpha')] = alpha
                ret_dict[(nx,ny,'alpha_std')] = alpha_std

        return ret_dict
    def differential_image_single(self):
        pass

if __name__ == "__main__":
    # Example usage
    diff_image = differential_image()
    source_used = pd.merge(diff_image.data_loader.reference_star_aper_df,diff_image.data_loader.source_df,on ='source_id',how='left')
    source_used = source_used[(source_used['group_quantile']<5)&(source_used['is_reference'])]
    diff_image.set_template("/home/test/workspace/Tianyu_pipeline/algorithm/data/testoutput/image/stacked_science/stacked_KOI68_3.fit", source_used)
    result = diff_image.differential_image_algo("/home/test/workspace/Tianyu_pipeline/algorithm/data/testoutput/image/calibrated_science/KOI68-0149_3_calibrated.fit")
    print(result)