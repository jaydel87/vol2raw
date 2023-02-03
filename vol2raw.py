#!/data/projects/xni/staff/jlivingstone/vol2raw/pyenv/bin/python3

import argparse
import numpy as np
import multiprocessing as mp
from functools import partial

def parse_arguments():
	argparser = argparse.ArgumentParser(description='Converts tomographic volumes (.vol) to .raw files.')
	argparser.add_argument('--volume', type=str, help='Name of file to convert.')
	argparser.add_argument('--bitdepth', type=str, default="default", help='Optional: desired bit depth of converted file. Default is the same as the original file.')
	#argparser.add_argument('--cpus', type=int, default=1, help='Optional: number of cpus to run on. Defaults to no threading.')

	return argparser.parse_args()
	

class raw:
	def __init__(self, args):
		
		dt = {"default" : "float32", "32bits" : "int32", "16bits" : "int16", "8bits" : "int8"}
		intbits = {"32bits" : 32, "16bits" : 16, "8bits" : 8}
				
		self.volName = args.volume
		infoFile = args.volume + ".info"
		self.info = self.get_vol_info(infoFile)
		self.slices = int(self.info["NUM_Z"])
		self.rows = int(self.info["NUM_Y"])
		self.cols = int(self.info["NUM_X"])
		self.vmin = float(self.info["ValMin"])
		self.vmax = float(self.info["ValMax"])
		self.bits = intbits[args.bitdepth]
		self.fromType = dt["default"]
		self.toType = dt[args.bitdepth]
		
		suffix = dt[args.bitdepth] + "_" + str(self.cols) + "x" + str(self.rows) + "x" + str(self.slices) + ".raw"
		
		self.saveFile = args.volume.replace(".vol", suffix).split("/")[-1]
		self.memmap = np.memmap(self.saveFile, dtype=self.toType , mode='w+', shape=(self.slices*self.rows*self.cols))


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
	
	
	def convert(self):
		self.orig = self.orig.copy()
		self.orig[self.orig == 0] = self.vmin - 1
		self.orig = ((self.orig - self.vmin) / (self.vmax - self.vmin)) * (2 ** self.bits - 1)

	def write_to_raw(self, sliceID):
		process = mp.current_process()
		print("Processing slice: " + str(sliceID) + " of " + str(self.slices))
		sliceSize = self.rows*self.cols
		start = sliceID*sliceSize
		stop = start + sliceSize
		startByte = sliceID*sliceSize*4 # 8 bits ins a byte, therefore 4 bytes in 32 bits
		self.orig = np.memmap(self.volName, dtype=self.fromType, mode='r', offset=startByte, shape=(sliceSize))
		
		if self.toType != "default":
			self.convert()
			self.memmap[start:stop] = self.orig[:]
		else:
			self.memmap[start:stop] = self.orig[:]
			self.orig.flush()


	def close(self):
		self.memmap.flush()
	

def main():
	args = parse_arguments()
	r = raw(args)
	slices = list(range(r.slices))
	for i in slices:
		r.write_to_raw(i)

	r.close()


if __name__ == '__main__':
    main()
