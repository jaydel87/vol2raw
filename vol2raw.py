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
	argparser.add_argument('--bitdepth', type=str, default="pyhst", help='Optional: desired bit depth of converted file. Default is the same as the pyhst output (32-bit float).')
	argparser.add_argument('--rescale', type=int, default=None, help='Optional: .')
	argparser.add_argument('--output', type=str, default="./", help='Optional: Path to store raw volumes. Default is current directory.')

	return argparser.parse_args()
	

class ht_vol:
	def __init__(self, args):
		
		self.pi = Pi2()
		
		dt = {"pyhst" : "float32", "32bits" : "uint32", "16bits" : "uint16", "8bits" : "uint8"}
				
		self.volName = args.volume
		infoFile = args.volume + ".info"
		self.info = self.get_vol_info(infoFile)
		self.slices = int(self.info["NUM_Z"])
		self.rows = int(self.info["NUM_Y"])
		self.cols = int(self.info["NUM_X"])
		self.voxelSize = float(self.info["voxelSize"])
		self.vmin = float(self.info["ValMin"])
		self.vmax = float(self.info["ValMax"])
		self.byteorder = self.info["BYTEORDER"].strip()
		
		self.fromType = "float32"
		self.toType = dt[args.bitdepth]
		
		self.rescaleTo = float(args.rescale)
		if self.rescaleTo != None:
			self.zoom = self.voxelSize / (self.rescaleTo / 1000.)
		else:
			self.zoom = 1
		
		strbits = args.bitdepth
		
		self.savePath = args.output
		self.saveFile = args.volume.replace(".vol", strbits).split("/")[-1]


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
		
		self.tmp = self.orig[:].copy()
		del self.orig
		
		if self.toType != "float32":
			self.convert()
		
		self.flag = "slice"
			
			
	def read_vol(self):
		print("Processing volume " + self.volName)
		self.orig = np.memmap(self.volName, dtype=self.fromType, mode='r', offset=0, shape=(self.slices*self.rows*self.cols))
		
		self.tmp = self.orig[:].copy()
		del self.orig
		
		if self.toType != "float32":
			self.convert()

		self.flag = "vol"

	
	def pi_image(self):
		self.x = round(self.cols * self.zoom)
		self.y = round(self.rows * self.zoom)
		self.z = round(self.slices * self.zoom)
		
		self.shape = str(self.y) + "x" + str(self.x) + "x" + str(self.z)
		
		pi_cmd = {"float32" : self.pi.newimage(ImageDataType.FLOAT32, self.y, self.x, self.z), 
					"uint32" : self.pi.newimage(ImageDataType.UINT32, self.y, self.x, self.z), 
					"uint16" : self.pi.newimage(ImageDataType.UINT16, self.y, self.x, self.z), 
					"uint8" : self.pi.newimage(ImageDataType.UINT8, self.y, self.x, self.z)}
					
		self.pi_img = pi_cmd[self.toType]
		
	
	def convert(self):
		intbits = {"uint32" : 32, "uint16" : 16, "uint8" : 8}
		print("Converting to " + str(intbits[self.toType]) + " bits.")
		#self.orig = self.orig.copy()
		self.vmin -= 1 / (2 ** intbits[self.toType] - 1)
		self.tmp[self.tmp == 0] = self.vmin # Zero-valued pixels (eg. pixels outside the reconstructed cylinder) should retain their zero-value after the conversion
		self.tmp[self.tmp >= self.vmax] = self.vmax
		self.tmp[self.tmp <= self.vmin] = self.vmin
		self.tmp = ((self.tmp - self.vmin) / (self.vmax - self.vmin)) * (2 ** intbits[self.toType] - 1)
		self.tmp = self.tmp.astype(self.toType)
		
	
	def reshape_img(self, slices, rows, cols):
		print("Reshaping volume from numPy (slices, rows, cols) format to pi2 format (rows, cols, slices).")
		self.tmp = self.tmp.astype(self.toType)
		self.tmp = self.tmp.reshape(slices, rows, cols)
		self.tmp = np.transpose(self.tmp, (1, 2, 0))
		

	def rescale_img(self):
		print("Rescaling to " + str(self.rescaleTo) + " nm pixel/voxel size.")
		self.pi.scale(self.tmp, self.pi_img, [0,0,0], False, "Nearest")
		self.tmp = self.pi_img.get_data()
		self.saveFile += "_rescaled_"+str(self.rescaleTo) + "nm"
		
	
	def reslice_img(self, direction):
		"""
		Direction = top, bottom, left or right
		"""
		pi.reslice(self.tmp, self.pi_img, direction)
		self.tmp = self.pi_img.get_data()
		self.saveFile += "_resliced_"+direction
				
		
	def write_to_raw(self):
		print("Writing to " + str(self.savePath) + str(self.saveFile) + "_" + self.shape + ".raw")
		self.pi_img.set_data(self.tmp)
		del self.tmp
		
		if self.flag == "vol":
			self.pi.writeraw(self.pi_img, self.savePath+self.saveFile)
		elif self.flag == "slice":
			self.pi.writerawblock(self.pi_img, self.savePath+self.saveFile, [0, 0, self.sliceID], [self.cols, self.rows, self.slices]) 
			
		self.write_raw_info()
		
						
	def write_raw_info(self):
		dims = self.pi_img.get_dimensions()
		infoFile = self.savePath + self.saveFile + "_" + str(dims[0]) + "x" + str(dims[1]) + "x" +  str(dims[2]) + ".raw.info"
		
		with open(infoFile, 'w') as f: 
			f.write("NUM_X = " + str(dims[0]) + "\n")
			f.write("NUM_Y = " + str(dims[1]) + "\n")
			f.write("NUM_Z = " + str(dims[2]) + "\n")
			f.write("voxelSize = " + str(self.voxelSize / self.zoom) + "\n")
			f.write("BYTEORDER = " + str(self.byteorder) + "\n")
			f.write("ValMin = " + str(self.vmin) + "\n")
			f.write("ValMax = " + str(self.vmax) + "\n")
		
		
	def vol_to_raw(self):
		self.read_vol()
		self.reshape_img(self.slices, self.rows, self.cols)
		self.pi_image()
		if self.zoom != 1:
			self.rescale_img()
		self.write_to_raw()
		
		
	def slice_to_raw(self, sliceID, totalSlices):
		self.slices = totalSlices
		self.read_slice(sliceID)
		self.reshape_img(1, self.rows, self.cols)
		self.pi_image()
		if self.zoom != 1:
			self.magnify()
		self.write_to_raw()
	

def main():
	args = parse_arguments()
	r = ht_vol(args)
	r.vol_to_raw()

if __name__ == '__main__':
    main()
