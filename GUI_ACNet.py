'''
AC-Net: Atmospheric correction neural Network model
Author: La Thi Oanh
Time: August 2021
Universtity: National Cheng Kung University-NCKU, Taiwan
'''


import numpy as np
import pandas as pd
from sklearn import preprocessing
import gdal
import fiona
import rasterio.mask
import rasterio as rio
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from tkinter import *
from tkinter import messagebox
import time
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, Dense, Dropout, Flatten, Reshape
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.regularizers import l2
import kerastuner as kt
import matplotlib.pyplot as plt
import os
import pathlib
from PIL import ImageTk, Image
import matplotlib as mpl


root = Tk()
root.title('AC-Net')
root.iconbitmap('ACNet.ico')
root.geometry("970x600")


## tell message when process complete
class BackendProcess:
    def __init__(self):
        self.finished = False

    def run(self):
        time.sleep(10)
        self.finished = True


## main class
class ACNet_app(object):
    def __init__(self, root):
        self.root = root
        self.labelframe = LabelFrame(self.root, text='Image inputs', foreground="blue", labelanchor='n', padx=10, pady=10, relief=GROOVE,
                                     borderwidth=2)
        self.labelframe.grid(row=0, column=0)
        self.mtl_label = Label(self.labelframe, text="Choose MTL file").grid(row=1, column=0)
        self.mtl_button = Button(self.labelframe, text="Select", command=self.openMTL).grid(row=1, column=2)
        self.mtl_scrollbar = Scrollbar(self.labelframe, orient='horizontal')
        self.entry_var_mtl = StringVar()
        self.mtl_entry = Entry(self.labelframe, textvariable=self.entry_var_mtl, width=50, borderwidth=2)
        self.mtl_entry.config(xscrollcommand=self.mtl_scrollbar.set)
        self.mtl_scrollbar['command'] = self.mtl_entry.xview
        self.mtl_entry.grid(row=1, column=1)
        self.mtl_scrollbar.grid(row=2, column=1, columnspan=1, sticky='ew')

        self.bands_label = Label(self.labelframe, text="Choose band list").grid(row=3, column=0)
        self.bands_button = Button(self.labelframe, text="Select", command=self.openbandlist).grid(row=3, column=2)
        self.band_scrollbar = Scrollbar(self.labelframe, orient='horizontal')
        self.bands = StringVar()
        self.band_listbox = Listbox(self.labelframe, listvariable=self.bands, width=50, height=8, borderwidth=2)
        self.band_listbox.config(xscrollcommand=self.band_scrollbar.set)
        self.band_scrollbar['command'] = self.band_listbox.xview
        self.band_listbox.grid(row=3, column=1)
        self.band_scrollbar.grid(row=4, column=1, columnspan=1, sticky='ew')

        self.angles_label = Label(self.labelframe, text="Choose angles files").grid(row=5, column=0)
        self.angles_button = Button(self.labelframe, text="Select", command=self.openangles).grid(row=5, column=2)
        self.angles_scrollbar = Scrollbar(self.labelframe, orient='horizontal')
        self.angles = StringVar()
        self.angles_listbox = Listbox(self.labelframe, listvariable=self.angles, width=50, height=4, borderwidth=2)
        self.angles_listbox.config(xscrollcommand=self.angles_scrollbar.set)
        self.angles_scrollbar['command'] = self.angles_listbox.xview
        self.angles_listbox.grid(row=5, column=1)
        self.angles_scrollbar.grid(row=6, column=1, columnspan=1, sticky='ew')

        self.aot_label = Label(self.labelframe, text="Choose aerosol file").grid(row=7, column=0)
        self.aot_button = Button(self.labelframe, text="Select", command=self.openAOT).grid(row=7, column=2)
        self.aot_scrollbar = Scrollbar(self.labelframe, orient='horizontal')
        self.entry_var_aot = StringVar()
        self.aot_entry = Entry(self.labelframe, textvariable=self.entry_var_aot, width=50, borderwidth=2)
        self.aot_entry.config(xscrollcommand=self.aot_scrollbar.set)
        self.aot_scrollbar['command'] = self.aot_entry.xview
        self.aot_entry.grid(row=7, column=1)
        self.aot_scrollbar.grid(row=8, column=1, columnspan=1, sticky='ew')

        self.labelframe2 = LabelFrame(self.root, text='Lake shapefile input', foreground="blue", labelanchor='n', padx=15, pady=10,
                                      relief=GROOVE, borderwidth=2)
        self.labelframe2.grid(row=1, column=0)
        self.shp_label1 = Label(self.labelframe2, text="Choose shapefile").grid(row=1, column=0)
        self.shp_button = Button(self.labelframe2, text="Select", command=self.openSHP).grid(row=1, column=2)
        self.shp_scrollbar = Scrollbar(self.labelframe2, orient='horizontal')
        self.entry_var_shp = StringVar()
        self.shp_entry = Entry(self.labelframe2, textvariable=self.entry_var_shp, width=50, borderwidth=2)
        self.shp_entry.config(xscrollcommand=self.shp_scrollbar.set)
        self.shp_scrollbar['command'] = self.shp_entry.xview
        self.shp_entry.grid(row=1, column=1)
        self.shp_scrollbar.grid(row=2, column=1, columnspan=1, sticky='ew')

        self.labelframe3 = LabelFrame(self.root, text='Image output', foreground="blue", labelanchor='n', padx=5, pady=10, relief=GROOVE,
                                      borderwidth=2)
        self.labelframe3.grid(row=2, column=0)
        self.output_label2 = Label(self.labelframe3, text="Choose output folder").grid(row=1, column=0)
        self.output_button = Button(self.labelframe3, text="Select", command=self.saveoutput).grid(row=1, column=2)
        self.output_scrollbar = Scrollbar(self.labelframe3, orient='horizontal')
        self.entry_var_output = StringVar()
        self.output_entry = Entry(self.labelframe3, textvariable=self.entry_var_output, width=50, borderwidth=2)
        self.output_entry.config(xscrollcommand=self.output_scrollbar.set)
        self.output_scrollbar['command'] = self.output_entry.xview
        self.output_entry.grid(row=1, column=1)
        self.output_scrollbar.grid(row=2, column=1, columnspan=1, sticky='ew')

        self.frame1 = Frame(self.root)
        self.frame1.grid(row=3, column=0)
        self.runAC_label = Label(self.frame1, text="Run AC-Net", foreground="blue").grid(row=1, column=1)
        self.run_button = Button(self.frame1, text="Run", command=self.runANN).grid(row=18, column=1)
        self.process = BackendProcess()

        self.frame2 = Frame(self.root)  # , borderwidth=2, relief="flat", width=500, height=560)
        self.frame2.grid(row=0, column=1, rowspan=6, sticky=NE)
        self.preview_label = Label(self.frame2, text="Output preview").grid(row=1, column=2, padx=50)
        self.preview_button = Button(self.frame2, text="Display", command=self.display).grid(row=1, column=3, pady=30)

        self.frame3 = Frame(self.root)
        self.frame3.grid(row=5, column=5, sticky=E)
        self.authorname = Label(self.frame3, text="By La Thi Oanh", font=('Bodoni MT', 7, 'italic')).grid(row=1, column=4, sticky=E)

    def openMTL(self):
        global metadata
        Root = tkinter.Tk()
        Root.withdraw()
        # open MTL file
        metadata = askopenfilename(title=u'Open MTL file', filetypes=[("MTL", ".txt")])
        self.entry_var_mtl.set(metadata)

    def openbandlist(self):
        global bandlist
        bandlist = filedialog.askopenfilenames(title='Choose band 1 to 7 and band 9 files', filetypes=[("TIF", ".tif")])
        self.bands.set(bandlist)

    def openangles(self):
        global angles_path
        angles_path = filedialog.askopenfilenames(title='Choose SAA, SZA, VAA, VZA files', filetypes=[("TIF", ".tif")])
        self.angles.set(angles_path)

    def openAOT(self):
        global AOT_path
        AOT_path = askopenfilename(title=u'Open sr_aerosol file', filetypes=[("TIF", ".tif")])
        self.entry_var_aot.set(AOT_path)

    def openSHP(self):
        global shp_path
        shp_path = askopenfilename(title=U'Open lake shapefile', filetypes=[("SHP", ".shp")])
        self.entry_var_shp.set(shp_path)

    def saveoutput(self):
        global save_path
        save_path = filedialog.askdirectory(title=U'Select output folder')
        self.entry_var_output.set(save_path)

    def runANN(self):
        global outpath3

        def get_metadata(metadata):
            fh = open(metadata)
            # Get rescaling parameters and sun_elevation angle
            mult_term = []
            add_term = []
            sun_elevation = float()
            for line in fh:
                # Read the file line-by-line looking for the reflectance transformation parameters
                if "REFLECTANCE_MULT_BAND_" in line:
                    mult_term.append(float(line.split("=")[1].strip()))
                elif "REFLECTANCE_ADD_BAND_" in line:
                    add_term.append(float(line.split("=")[1].strip()))
                elif "SUN_ELEVATION" in line:
                    # We're also getting the sun elevation from the metadata. It has

                    sun_elevation = float(line.split("=")[1].strip())
            fh.close()  # Be sure to close an open file

            return mult_term, add_term, sun_elevation

        [mult_term, add_term, sun_elevation] = get_metadata(metadata)

        def DN_toTOAref(mult_term, add_term, sun_elevation, bandlist):

            with rio.open(bandlist[0]) as src1:
                image_band1 = src1.read(1)

            image_masked_band1 = np.ma.masked_array(image_band1, mask=(image_band1 == 0))  # exclude 0 value
            constant = 0.01745329251994444444444444444444  # Constant is calculated (3.14/180) which is converting the sun-angle to sun_radians which was suggested by WOlfgang
            toa1 = (mult_term[0] * image_masked_band1.astype(float) + add_term[0])
            solar_z = np.cos((90 - float(sun_elevation)) * float(constant))
            toa_band1 = (toa1.astype(float) / solar_z)

            with rio.open(bandlist[1]) as src2:
                image_band2 = src2.read(1)
            image_masked_band2 = np.ma.masked_array(image_band2, mask=(image_band2 == 0))  # exclude 0 value
            constant = 0.01745329251994444444444444444444
            toa2 = (mult_term[1] * image_masked_band2.astype(float) + add_term[1])
            solar_z = np.cos((90 - float(sun_elevation)) * float(constant))
            toa_band2 = (toa2.astype(float) / solar_z)

            with rio.open(bandlist[2]) as src3:
                image_band3 = src3.read(1)
            image_masked_band3 = np.ma.masked_array(image_band3, mask=(image_band3 == 0))  # exclude 0 value
            constant = 0.01745329251994444444444444444444
            toa3 = (mult_term[2] * image_masked_band3.astype(float) + add_term[2])
            solar_z = np.cos((90 - float(sun_elevation)) * float(constant))
            toa_band3 = (toa3.astype(float) / solar_z)

            with rio.open(bandlist[3]) as src4:
                image_band4 = src4.read(1)
            image_masked_band4 = np.ma.masked_array(image_band4, mask=(image_band4 == 0))  # exclude 0 value
            constant = 0.01745329251994444444444444444444
            toa4 = (mult_term[3] * image_masked_band4.astype(float) + add_term[3])
            solar_z = np.cos((90 - float(sun_elevation)) * float(constant))
            toa_band4 = (toa4.astype(float) / solar_z)

            with rio.open(bandlist[4]) as src5:
                image_band5 = src5.read(1)
            image_masked_band5 = np.ma.masked_array(image_band5, mask=(image_band5 == 0))  # exclude 0 value
            constant = 0.01745329251994444444444444444444
            toa5 = (mult_term[4] * image_masked_band5.astype(float) + add_term[4])
            solar_z = np.cos((90 - float(sun_elevation)) * float(constant))
            toa_band5 = (toa5.astype(float) / solar_z)

            with rio.open(bandlist[5]) as src6:
                image_band6 = src6.read(1)
            image_masked_band6 = np.ma.masked_array(image_band6, mask=(image_band6 == 0))  # exclude 0 value
            constant = 0.01745329251994444444444444444444
            toa6 = (mult_term[5] * image_masked_band6.astype(float) + add_term[5])
            solar_z = np.cos((90 - float(sun_elevation)) * float(constant))
            toa_band6 = (toa6.astype(float) / solar_z)

            with rio.open(bandlist[6]) as src7:
                image_band7 = src7.read(1)
            image_masked_band7 = np.ma.masked_array(image_band7, mask=(image_band7 == 0))  # exclude 0 value
            constant = 0.01745329251994444444444444444444
            toa7 = (mult_term[6] * image_masked_band7.astype(float) + add_term[6])
            solar_z = np.cos((90 - float(sun_elevation)) * float(constant))
            toa_band7 = (toa7.astype(float) / solar_z)

            with rio.open(bandlist[7]) as src8:
                image_band8 = src8.read(1)
            image_masked_band8 = np.ma.masked_array(image_band8, mask=(image_band8 == 0))  # exclude 0 value
            constant = 0.01745329251994444444444444444444
            toa8 = (mult_term[7] * image_masked_band8.astype(float) + add_term[7])
            solar_z = np.cos((90 - float(sun_elevation)) * float(constant))
            toa_band8 = (toa8.astype(float) / solar_z)

            return toa_band1, toa_band2, toa_band3, toa_band4, toa_band5, toa_band6, toa_band7, toa_band8

        [toa_band1, toa_band2, toa_band3, toa_band4, toa_band5, toa_band6, toa_band7, toa_band8] = \
            DN_toTOAref(mult_term, add_term, sun_elevation, bandlist)

        ## CALCULATE RELATIVE AZIMUTH ANGLE FROM SUN AZIMUTH AND SENSOR AZIMUTH ANGLE
        def RAA(angles_path):  # SAA, SZA, VAA, VZA
            ## read raster image as array
            SAA = gdal.Open(angles_path[0]).ReadAsArray()
            VAA = gdal.Open(angles_path[2]).ReadAsArray()
            sun_azi_angle = SAA / 100
            sen_azi_angle = VAA / 100
            difference_value = abs(sun_azi_angle - sen_azi_angle)  # abs(Sensor Azimuth - 180.0 - Solar azimuth)
            dif_row = difference_value.shape[0]
            dif_col = difference_value.shape[1]
            RAA = np.zeros([dif_row, dif_col])
            for i in range(dif_row):
                for j in range(dif_col):
                    if difference_value[i, j] > 180.0:
                        RAA[i, j] = 360.0 - difference_value[i, j]
                    elif difference_value[i, j] == 0:
                        RAA[i, j] = 0.0
                    else:
                        RAA[i, j] = 180.0 - difference_value[i, j]
            return RAA

        RAA = RAA(angles_path)

        ## calculating cosine of RAA angle
        def cosRAA(RAA):
            RAA[RAA == 0] = np.nan  # convert 0 value to nan to avoid calculate cosine for pixel have 0 value
            cos_RAA = np.cos(RAA)
            cos_RAA[np.isnan(cos_RAA)] = 0  # convert nan back to 0 value
            cos_RAA = np.float32(cos_RAA)  # convert float64 to float32
            return cos_RAA

        cos_RAA = cosRAA(RAA)

        def cos_SZA_VZA(angles_path):
            ## read raster image as array
            SZA = gdal.Open(angles_path[1]).ReadAsArray()
            VZA = gdal.Open(angles_path[3]).ReadAsArray()
            sun_ze_angle = SZA / 100
            sen_ze_angle = VZA / 100
            sun_ze_angle[
                sun_ze_angle == 0] = np.nan  # convert 0 value to nan to avoid calculate cosine for pixel have 0 value
            cos_SZA = np.cos(sun_ze_angle)
            cos_SZA[np.isnan(cos_SZA)] = 0  # convert nan back to 0 value
            cos_SZA_final = np.float32(cos_SZA)  # convert float64 to float32
            sen_ze_angle[
                sen_ze_angle == 0] = np.nan  # convert 0 value to nan to avoid calculate cosine for pixel have 0 value
            cos_VZA = np.cos(sen_ze_angle)
            cos_VZA[np.isnan(cos_VZA)] = 0  # convert nan back to 0 value
            cos_VZA_final = np.float32(cos_VZA)  # convert float64 to float32
            return cos_SZA_final, cos_VZA_final

        [cos_SZA_final, cos_VZA_final] = cos_SZA_VZA(angles_path)

        def AOT(AOT_path):
            AOT_array = gdal.Open(AOT_path).ReadAsArray()
            AOT_array[AOT_array == 1] = 0  # convert nan back to 0 value
            AOT = AOT_array * 0.0001  ## recaling for 10000
            AOT = np.float32(AOT)  # convert float64 to float32
            return AOT

        AOT = AOT(AOT_path)

        def mergebands():
            toab1 = np.array(toa_band1, dtype='float32')
            toab1[toab1 == 2.e-05] = 0
            toab2 = np.array(toa_band2, dtype='float32')
            toab2[toab2 == 2.e-05] = 0
            toab3 = np.array(toa_band3, dtype='float32')
            toab3[toab3 == 2.e-05] = 0
            toab4 = np.array(toa_band4, dtype='float32')
            toab4[toab4 == 2.e-05] = 0
            toab5 = np.array(toa_band5, dtype='float32')
            toab5[toab5 == 2.e-05] = 0
            toab6 = np.array(toa_band6, dtype='float32')
            toab6[toab6 == 2.e-05] = 0
            toab7 = np.array(toa_band7, dtype='float32')
            toab7[toab7 == 2.e-05] = 0
            toab8 = np.array(toa_band8, dtype='float32')
            toab8[toab8 == 2.e-05] = 0

            filelist = np.concatenate((toab1, toab2, toab3, toab4, toab5, toab6, toab7, toab8,
                                       cos_VZA_final, cos_SZA_final, cos_RAA, AOT), axis=0)
            filelist_reshape = filelist.reshape(12, toab1.shape[0], toab1.shape[1]).astype('float32')

            ## SAVING to TIF image
            with rio.open(bandlist[0]) as src:  # choose one image
                ras_data = src.read()
                ras_meta = src.meta
            ras_meta.update(count=len(filelist_reshape))
            ras_meta['dtype'] = "float32"
            ras_meta['No Data'] = 0.0
            filename = 'TOA_Angles_AOT'
            suffix = '.tif'
            out_path = Path(save_path, filename).with_suffix(suffix)
            with rio.open(out_path, 'w', **ras_meta) as dst:
                dst.write(filelist_reshape)
            return out_path

        out_path = mergebands()

        def clip_shp(shp_path, out_path):
            global out_clip0, clippath
            with fiona.open(shp_path, "r") as shapefile:
                shapes = [feature["geometry"] for feature in shapefile]
            with rio.open(out_path) as src1:
                out_image, out_transform = rio.mask.mask(src1, shapes=shapes, crop=True)
                out_meta = src1.meta
                out_meta['dtype'] = "float32"
                out_meta['No Data'] = 0.0
                out_meta.update({"driver": "GTiff",
                                 "height": out_image.shape[1],
                                 "width": out_image.shape[2],
                                 "transform": out_transform})
            clip_name0 = 'TOA_Angles_AOT_clip'
            suffix = '.tif'
            out_clip0 = Path(save_path, clip_name0).with_suffix(suffix)
            with rio.open(out_clip0, "w", **out_meta) as dst:
                dst.write(out_image)
            return out_clip0

        out_clip0 = clip_shp(shp_path, out_path)
        p = pathlib.PureWindowsPath(out_clip0)  # convert windowsPath to string
        clippath = str(p.as_posix())

        def correct_input(clippath):
            global outpath, clippath1
            img = gdal.Open(clippath).ReadAsArray()
            aver_toab8 = np.average(img[7])
            toab8_aver = np.zeros([img[7].shape[0], img[7].shape[1]], dtype='float32')
            for i in range(img[7].shape[0]):
                for j in range(img[7].shape[1]):
                    toab8_aver[i, j] = aver_toab8

            aver_cosRAA = np.average((img[10]))
            cosRAA_aver = np.zeros([img[10].shape[0], img[10].shape[1]], dtype='float32')
            for i in range(img[10].shape[0]):
                for j in range(img[10].shape[1]):
                    cosRAA_aver[i, j] = aver_cosRAA

            filelist_corr = np.concatenate((img[0], img[1], img[2], img[3], img[4], img[5], img[6], toab8_aver,
                                            img[8], img[9], cosRAA_aver, img[11]), axis=0)
            filelist_reshape1 = filelist_corr.reshape(12, img[0].shape[0], img[0].shape[1]).astype('float32')

            ## SAVING to TIF image
            with rio.open(clippath) as src:  # choose one image
                ras_data = src.read()
                ras_meta = src.meta
            ras_meta.update(count=len(filelist_reshape1))
            ras_meta['dtype'] = "float32"
            ras_meta['No Data'] = 0.0
            filename = 'TOA_Angles_AOT_clip_corr'
            suffix = '.tif'
            outpath = Path(save_path, filename).with_suffix(suffix)
            with rio.open(outpath, 'w', **ras_meta) as dst:
                dst.write(filelist_reshape1)
            return outpath

        outpath = correct_input(clippath)
        p1 = pathlib.PureWindowsPath(outpath)  # convert windowsPath to string
        clippath1 = str(p1.as_posix())

        ## PATCH GENERATE 3*3 pixels
        def goodpixel(clippath1):
            global good_pix, sel_pix_row_i, sel_pix_col_j, n_bands, data_img
            data_img = gdal.Open(clippath1).ReadAsArray()
            n_bands, n_row, n_col = data_img.shape
            good_pix = np.zeros([n_bands, 3, 3], dtype='float32')
            m = 0
            sel_pix_row_i = []
            sel_pix_col_j = []
            for i in range(1, (n_row - 2)):
                for j in range(1, (n_col - 2)):
                    a1 = i - 1
                    a2 = i + 2
                    b1 = j - 1
                    b2 = j + 2
                    for z in range(n_bands):
                        good_pix[z, ...] = data_img[z, a1:a2, b1:b2]
                    m = m + 1  # m: NUMBER OF GOOD PIXELS
                    sel_pix_row_i += [i]  # sel_pix_row_i: ROW I OF GOOD PIXEL
                    sel_pix_col_j += [j]  # sel_pix_col_j: COL J OF GOOD PIXEL

            return good_pix, sel_pix_row_i, sel_pix_col_j, n_bands, data_img

        [good_pix, sel_pix_row_i, sel_pix_col_j, n_bands, data_img] = goodpixel(clippath1)

        def extract_patches(sel_pix_row_i, sel_pix_col_j):
            global patches_4D, n_patch
            sel_pix_row_i = np.asarray(sel_pix_row_i)
            sel_pix_col_j = np.asarray(sel_pix_col_j)
            n_patch = sel_pix_row_i.shape[0]  # Number of patches
            patches_4D = np.zeros([n_patch, n_bands, 3, 3],
                                  dtype='float32')  # patches in 4 dimensions (n_patch, n_band, 3, 3)
            for i in range(n_patch):
                a1 = sel_pix_row_i[i] - 1
                a2 = sel_pix_row_i[i] + 2
                b1 = sel_pix_col_j[i] - 1
                b2 = sel_pix_col_j[i] + 2
                for z in range(n_bands):
                    patches_4D[i, z, ...] = data_img[z, a1:a2, b1:b2]
            return patches_4D, n_patch

        patches_4D, n_patch = extract_patches(sel_pix_row_i, sel_pix_col_j)

        def Xtest_generate(patches_4D):
            global TOA_xtesting_reshape, angles_xtesting_final, AOT_xtesting_reshape
            xtesting = patches_4D
            n_testsamples = xtesting.shape[0]
            TOA_xtesting = np.zeros([n_testsamples, 8, 3, 3])
            for i in range(0, n_testsamples):
                for j in range(0, 8):
                    TOA_xtesting[i][j] = xtesting[i][j]
            TOA_xtesting_reshape = TOA_xtesting.reshape(TOA_xtesting.shape[0], 3, 3, 8, 1).astype('float32')

            angles_xtesting = np.zeros([n_testsamples, 3, 3, 3])
            for m in range(0, n_testsamples):
                for k in range(0, 3):
                    angles_xtesting[m][k] = xtesting[m][k + 8]
            angles_xtesting_reshape = angles_xtesting.reshape(angles_xtesting.shape[0], 3 * 3 * 3).astype('float32')

            ## normalize angles data from (-1,1) to scale (0,1)
            pdread_angles_test = pd.DataFrame(angles_xtesting_reshape)
            scaler2 = preprocessing.MinMaxScaler()
            colums2 = pdread_angles_test.columns
            transform2 = scaler2.fit_transform(pdread_angles_test)
            angles_xtesting_normalize = pd.DataFrame(transform2, columns=colums2)
            angles_xtesting_normalize.head()
            angles_xtesting_final = angles_xtesting_normalize.to_numpy()

            AOT_xtesting = np.zeros([n_testsamples, 1, 3, 3])
            for n in range(0, n_testsamples):
                for h in range(0, 1):
                    AOT_xtesting[n][h] = xtesting[n][h + 11]
            AOT_xtesting_reshape = AOT_xtesting.reshape(AOT_xtesting.shape[0], 3 * 3 * 1).astype('float32')
            return TOA_xtesting_reshape, angles_xtesting_final, AOT_xtesting_reshape

        TOA_xtesting_reshape, angles_xtesting_final, AOT_xtesting_reshape = Xtest_generate(patches_4D)

        def load_trainingdata():
            global TOA_xtrain, angles_xtrain, AOT_xtrain, ytrain_iCOR, TOA_xvali, angles_xvali, AOT_xvali, y_vali_iCOR
            TOA_xtrain = np.load('TOA_XTrain.npy')
            angles_xtrain = np.load('angles_XTrain.npy')
            AOT_xtrain = np.load('AOT_XTrain.npy')
            ytrain_iCOR = np.load('iCOR_YTrain.npy')

            TOA_xvali = np.load('TOA_XVali.npy')
            angles_xvali = np.load('angles_XVali.npy')
            AOT_xvali = np.load('AOT_XVali.npy')
            y_vali_iCOR = np.load('iCOR_Y_vali.npy')

            return TOA_xtrain, angles_xtrain, AOT_xtrain, ytrain_iCOR, TOA_xvali, angles_xvali, AOT_xvali, y_vali_iCOR

        TOA_xtrain, angles_xtrain, AOT_xtrain, ytrain_iCOR, TOA_xvali, angles_xvali, AOT_xvali, y_vali_iCOR = load_trainingdata()

        def build_model(hp):
            global hypermodel
            ## 3 Build a neural network model (Keras Functional API for Multiple inputs and mixed data)
            TOA_input = Input((3, 3, 8, 1))
            angles_input = Input((27,))
            AOT_input = Input((9,))

            # Create 3CN layers  ("valid" = without padding; "same" = with zero padding)
            x = Conv3D(filters=16, kernel_size=(1, 1, 3), strides=(1, 1, 1), padding='valid', activation='relu',
                       kernel_regularizer=l2(0.01), activity_regularizer=l2(0.01))(TOA_input)
            x = Conv3D(filters=16, kernel_size=(1, 1, 3), strides=(1, 1, 1), padding='valid', activation='relu',
                       kernel_regularizer=l2(0.01), activity_regularizer=l2(0.01))(x)
            x = Conv3D(filters=16, kernel_size=(1, 1, 3), strides=(1, 1, 1), padding='valid', activation='relu',
                       kernel_regularizer=l2(0.01), activity_regularizer=l2(0.01))(x)
            x = Flatten()(x)  # output shape =
            x = Model(inputs=TOA_input, outputs=x)
            x.summary()
            x = Reshape((288,))(x.output)
            # Combine x (TOA_extracted from CNN layers) and Angles data and AOT data
            combined = Concatenate(axis=1)([x, angles_input, AOT_input])

            ## Now apply Fully connected layers and then prediction on the combined data
            y = Dense(324, activation='relu')(combined)
            y = Dropout(hp.Float("dropout1", 0, 0.5, step=0.1, default=0.5))(y)
            y = Dense(hp.Int("hidden1_size", 300, 400, step=10), activation='relu')(y)
            y = Dropout(hp.Float("dropout2", 0, 0.5, step=0.1, default=0.5))(y)
            y = Dense(hp.Int("hidden2_size", 200, 300, step=10), activation='relu')(y)  # activation='relu'
            y = Dropout(hp.Float("dropout3", 0, 0.5, step=0.1, default=0.5))(y)
            y = Dense(hp.Int("hidden3_size", 100, 200, step=10), activation='sigmoid')(y)
            y = Dropout(hp.Float("dropout4", 0, 0.5, step=0.1, default=0.5))(y)
            y = Dense(hp.Int("hidden4_size5", 20, 100, step=10), activation='linear')(y)
            y = Dropout(hp.Float("dropout5", 0, 0.5, step=0.1, default=0.5))(y)
            y = Dense(5, activation='sigmoid')(y)

            ## output
            model = Model(inputs=[TOA_input, angles_input, AOT_input], outputs=y)
            model.summary()

            # Compiling model
            model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(
                hp.Float("learning_rate", 1e-4, 1e-2)), metrics=['mape', 'accuracy'])
            return model

        ## tune number of epochs

        tuner = kt.Hyperband(build_model,
                             objective='val_accuracy',
                             max_epochs=30,
                             hyperband_iterations=2,
                             )
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        ## run the hyperparameter search. it be same as for model fit
        tuner.search([TOA_xtrain, angles_xtrain, AOT_xtrain], ytrain_iCOR,
                     validation_data=([TOA_xvali, angles_xvali, AOT_xvali], y_vali_iCOR),
                     callbacks=[early_stopping], batch_size=256, verbose=2)
        ## get the optimal hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print(f"""The hyperparameter search is complete. 
         The optimal number of units in the 1st hidden layer is {best_hps.get('hidden1_size')}; 
         The optimal number of units in the 2nd hidden layer is {best_hps.get('hidden2_size')};
         The optimal number of units in the 3rd hidden layer is {best_hps.get('hidden3_size')};
         The optimal number of units in the 4th hidden layer is {best_hps.get('hidden4_size5')};
         The optimal 1st dropout is {best_hps.get('dropout1')};
         The optimal 2nd dropout is {best_hps.get('dropout2')};
         The optimal 3rd dropout is {best_hps.get('dropout3')};
         The optimal 4th dropout is {best_hps.get('dropout4')};
         The optimal 5th dropout is {best_hps.get('dropout5')};
         The optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.""")

        ##TRAIN THE MODEL
        # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
        model = tuner.hypermodel.build(best_hps)
        train_history = model.fit([TOA_xtrain, angles_xtrain, AOT_xtrain], ytrain_iCOR,
                                  validation_data=([TOA_xvali, angles_xvali, AOT_xvali], y_vali_iCOR),
                                  batch_size=256, epochs=30, verbose=2)
        val_acc_per_epoch = train_history.history['val_loss']
        best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
        print('Best epoch: %d' % (best_epoch,))

        # after finding out the best no of epochs. Re-instantiate the hypermodel and train it with the optimal number of epochs from above.
        hypermodel = tuner.hypermodel.build(best_hps)
        # Retrain the model
        hypermodel.fit([TOA_xtrain, angles_xtrain, AOT_xtrain], ytrain_iCOR,
                       validation_data=([TOA_xvali, angles_xvali, AOT_xvali], y_vali_iCOR),
                       batch_size=256, epochs=best_epoch)

        y_prediction = hypermodel.predict([TOA_xtesting_reshape, angles_xtesting_final,
                                           AOT_xtesting_reshape])  # prediction from x_test using model above

        def resulttransfer(y_prediction):
            global out_path2, outpath2
            pred_B1 = y_prediction[:, 0]
            pred_B2 = y_prediction[:, 1]
            pred_B3 = y_prediction[:, 2]
            pred_B4 = y_prediction[:, 3]
            pred_B5 = y_prediction[:, 4]
            ## Reshape into image
            pred_B1_reshape = np.reshape(pred_B1, (1, -1))
            pred_B1_2d = np.reshape(pred_B1_reshape, (data_img.shape[1] - 3, data_img.shape[2] - 3), order='C')
            pred_B2_reshape = np.reshape(pred_B2, (1, -1))
            pred_B2_2d = np.reshape(pred_B2_reshape, (data_img.shape[1] - 3, data_img.shape[2] - 3), order='C')
            pred_B3_reshape = np.reshape(pred_B3, (1, -1))
            pred_B3_2d = np.reshape(pred_B3_reshape, (data_img.shape[1] - 3, data_img.shape[2] - 3), order='C')
            pred_B4_reshape = np.reshape(pred_B4, (1, -1))
            pred_B4_2d = np.reshape(pred_B4_reshape, (data_img.shape[1] - 3, data_img.shape[2] - 3), order='C')
            pred_B5_reshape = np.reshape(pred_B5, (1, -1))
            pred_B5_2d = np.reshape(pred_B5_reshape, (data_img.shape[1] - 3, data_img.shape[2] - 3), order='C')

            filelist2 = np.concatenate((pred_B1_2d, pred_B2_2d, pred_B3_2d, pred_B4_2d, pred_B5_2d), axis=0)
            filelist_reshape2 = filelist2.reshape(5, pred_B1_2d.shape[0], pred_B1_2d.shape[1]).astype('float32')

            ## SAVING to TIF image
            with rio.open(clippath1) as src2:  # choose one image
                out2_data = src2.read()
                out2_meta = src2.meta
            out2_meta.update(count=len(filelist_reshape2))
            out2_meta['dtype'] = "float32"
            out2_meta['No Data'] = 0.0
            filename2 = 'prediction'
            suffix = '.tif'
            out_path2 = Path(save_path, filename2).with_suffix(suffix)
            with rio.open(out_path2, 'w',
                          **out2_meta) as dst:  # write image with the same shape with the selected image aboved
                dst.write(filelist_reshape2)
            return out_path2

        out_path2 = resulttransfer(y_prediction)
        p2 = pathlib.PureWindowsPath(out_path2)  # convert windowsPath to string
        outpath2 = str(p2.as_posix())

        def clip_output(shp_path, outpath2):
            global out_path3, outpath3
            ## now clip the RGBprediction image
            with fiona.open(shp_path, "r") as shapefile:
                shapes = [feature["geometry"] for feature in shapefile]
            with rio.open(outpath2) as src3:
                out_image3, out_transform3 = rio.mask.mask(src3, shapes=shapes, crop=True)
                out_meta3 = src3.meta
                out_meta3['dtype'] = "float32"
                out_meta3['No Data'] = 0.0
                out_meta3.update({"driver": "GTiff",
                                  "height": out_image3.shape[1],
                                  "width": out_image3.shape[2],
                                  "transform": out_transform3})
            filename3 = 'Rrs'
            suffix = '.tif'
            out_path3 = Path(save_path, filename3).with_suffix(suffix)
            with rio.open(out_path3, 'w',
                          **out_meta3) as dst:  # write image with the same shape with the selected image aboved
                dst.write(out_image3)
            return out_path3

        out_path3 = clip_output(shp_path, out_path2)
        p3 = pathlib.PureWindowsPath(out_path3)  # convert windowsPath to string
        outpath3 = str(p3.as_posix())

        ## remove files (keep only final Rrs Rrs
        r1 = os.path.join(save_path, 'TOA_Angles_AOT.tif')
        r2 = os.path.join(save_path, 'TOA_Angles_AOT_clip.tif')
        r3 = os.path.join(save_path, 'TOA_Angles_AOT_clip_corr.tif')
        r4 = os.path.join(save_path, 'prediction.tif')
        os.remove(r1)
        os.remove(r2)
        os.remove(r3)
        os.remove(r4)


        def plot_result(outpath3):
            img = gdal.Open(outpath3).ReadAsArray()
            vmin = np.amin(img)
            vmax = np.amax(img)

            img[img == 0] = 'nan'
            Image1 = img[0]
            Image2 = img[1]
            Image3 = img[2]
            Image4 = img[3]
            Image5 = img[4]
            # create figure
            fig = plt.figure(figsize=(4.8, 4.5))

            # setting values to rows and column variables
            rows = 3
            columns = 2

            # Adds a subplot at the 1st position
            fig.add_subplot(rows, columns, 1)
            # showing image
            plt.imshow(Image1)
            plt.axis('off')
            plt.title("B1", fontsize=10)

            # Adds a subplot at the 2nd position
            fig.add_subplot(rows, columns, 2)
            # showing image
            plt.imshow(Image2)
            plt.axis('off')
            plt.title("B2", fontsize=10)

            # Adds a subplot at the 3rd position
            fig.add_subplot(rows, columns, 3)
            # showing image
            plt.imshow(Image3)
            plt.axis('off')
            plt.title("B3", fontsize=10)

            # Adds a subplot at the 4th position
            fig.add_subplot(rows, columns, 4)
            # showing image
            plt.imshow(Image4)
            plt.axis('off')
            plt.title("B4", fontsize=10)

            # Adds a subplot at the 5th position
            fig.add_subplot(rows, columns, 5)
            # showing image
            plt.imshow(Image5)
            plt.axis('off')
            plt.title("B5", fontsize=10)

            fig.subplots_adjust(right=0.7, left=0.1)
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            ax = fig.add_axes([0.75, 0.15, 0.03, 0.7])
            clb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='viridis'), cax=ax, orientation='vertical')
            clb.ax.set_title('Rrs(sr-1)', fontsize=10)

            # save to figure
            plt.savefig(os.path.join(save_path, 'Rrs_plot.PNG'))
            return

        plot_result(outpath3)

        self.process.run()
        messagebox.showinfo('Info', 'Process completed!')
    def display(self):
        self.im = Image.open(os.path.join(save_path, 'Rrs_plot.PNG'))
        self.photo = ImageTk.PhotoImage(self.im)
        self.display = Label(self.frame2, image=self.photo, bg='white', compound='bottom')
        self.display.grid(row=2, column=1, columnspan=5, rowspan=7)



if __name__ == '__main__':
    app = ACNet_app(root)
    root.mainloop()
