#!/data/projects/xni/staff/jlivingstone/vol2raw/pyenv/bin/python3

import sys
import argparse
import numpy as np
import multiprocessing as mp
from scipy import ndimage

sys.path.append('/data/id16a/inhouse1/sware/NRstitcher/pi2/bin-linux64/release-nocl/')

from pi2py2 import *

def parse_arguments():
	argparser = argparse.ArgumentParser(description='Converts tomographic volumes (.vol) to .raw files.')
	argparser.add_argument('--volume', type=str, help='Name of file to convert.')
	argparser.add_argument('--bitdepth', type=str, default="default", help='Optional: desired bit depth of converted file. Default is the same as the original file.')

	return argparser.parse_args()
	

class ht_vol:
	def __init__(self, args):
		
		self.pi = Pi2()
		
		dt = {"default" : "float32", "32bits" : "uint32", "16bits" : "uint16", "8bits" : "uint8"}
				
		self.volName = args.volume
		infoFile = args.volume + ".info"
		self.info = self.get_vol_info(infoFile)
		self.slices = int(self.info["NUM_Z"])
		self.rows = int(self.info["NUM_Y"])
		self.cols = int(self.info["NUM_X"])
		self.voxelSize = float(self.info["voxelSize"])
		self.vmin = float(self.info["ValMin"])
		self.vmax = float(self.info["ValMax"])
		self.shape = str(self.rows) + "x" + str(self.cols) + "x" + str(self.slices)
		
		self.fromType = "float32"
		self.toType = dt[args.bitdepth]
		
		self.zoom = 1
		
		strbits = args.bitdepth
		
		self.saveFile = args.volume.replace(".vol", strbits).split("/")[-1]
		
		print("\n"+str(self.volName))


	def get_vol_info(self, infoFile):
		info = {}
		with open(infoFile) as f:
			for line in f.readlines():
				try:
					key, value = line.replace(" ", "").split("=")
					info[key] = value
				except:
					continue
				
		return info
		
		
	def read_slice(self, sliceID):
		print("Processing slice: " + str(sliceID) + " of " + str(self.slices))
		self.sliceID = sliceID
		sliceSize = self.rows*self.cols
		start = sliceID*sliceSize
		stop = start + sliceSize
		startByte = sliceID*sliceSize*4 # 8 bits ins a byte, therefore 4 bytes in 32 bits
		self.orig = np.memmap(self.volName, dtype=self.fromType, mode='r', offset=startByte, shape=(sliceSize))
		
		if self.toType != "float32":
			self.convert()
		
		self.new = self.orig[:]
		self.reshape_img(1, self.rows, self.cols)
		self.flag = "slice"
			
			
	def read_vol(self):
		print("Processing volume.")
		self.orig = np.memmap(self.volName, dtype=self.fromType, mode='r', offset=0, shape=(self.slices*self.rows*self.cols))
		
		if self.toType != "float32":
			self.convert()

		self.new = self.orig[:]
		self.reshape_img(self.slices, self.rows, self.cols)
		self.flag = "vol"

	
	def convert(self):
		intbits = {"uint32" : 32, "uint16" : 16, "uint8" : 8}
		print("Converting to " + str(intbits[self.toType]) + " bits.")
		self.orig = self.orig.copy()
		self.orig[self.orig == 0] = self.vmin - 1 # Zero-valued pixels (eg. pixels outside the reconstructed cylinder) should retain their zero-value after the conversion
		self.orig = ((self.orig - self.vmin) / (self.vmax - self.vmin)) * (2 ** intbits[self.toType] - 1)
		self.orig = self.orig.astype(self.toType)
		
	
	def reshape_img(self, slices, rows, cols):
		print("Reshaping image data.")
		self.new = self.new.astype(self.toType)
		self.new = self.new.reshape(slices, rows, cols)
		self.new = np.transpose(self.new, (1, 2, 0))
		

	def magnify(self):
		print("Magnifying image.")
		self.new = ndimage.zoom(self.new, self.zoom, order=2)
		self.rows, self.cols, self.slices = self.new.shape
		self.saveFile += str(int(self.voxelSize / self.zoom))
		

	def write_to_raw(self):
		print("Writing to ./" + str(self.saveFile) + "_" + self.shape + ".raw")
		pi_cmd = {"float32" : self.pi.newimage(ImageDataType.FLOAT32, self.cols, self.rows, self.slices), 
					"uint32" : self.pi.newimage(ImageDataType.UINT32, self.cols, self.rows, self.slices), 
					"uint16" : self.pi.newimage(ImageDataType.UINT16, self.cols, self.rows, self.slices), 
					"uint8" : self.pi.newimage(ImageDataType.UINT8, self.cols, self.rows, self.slices)}
					
		self.raw_img = pi_cmd[self.toType]
		self.raw_img.set_data(self.new)
		
		if self.flag == "vol":
			self.pi.writeraw(self.raw_img, self.saveFile)
		elif self.flag == "slice"
			self.pi.writerawblock(self.raw_img, self.saveFile, [0, 0, self.sliceID], [self.cols, self.rows, self.slices]) 
		
		
	def vol_to_raw(self):
		self.read_vol()
		if self.zoom > 1:
			self.magnify()
		self.write_to_raw()
		
		
	def slice_to_raw(self, sliceID, totalSlices):
		self.slices = totalSlices
		self.read_slice(sliceID)
		if self.zoom > 1:
			self.magnify()
		self.write_to_raw()
	

def main():
	args = parse_arguments()
	r = ht_vol(args)
	r.vol_to_raw()

if __name__ == '__main__':
    main()
