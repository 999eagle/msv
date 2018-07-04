#!../.venv/bin/python3
import sys
import argparse
import numpy as np
from scipy import fftpack, misc
import matplotlib.pyplot as plt
import skimage, skimage.io
import itertools
from enum import Enum
import struct

BLOCKSIZE = 8
QUANTIZATION_TABLE = np.array([
	[16, 11, 10, 16, 24, 40, 51, 61],
	[12, 12, 14, 19, 26, 58, 60, 55],
	[14, 13, 16, 24, 40, 57, 69, 56],
	[14, 17, 22, 29, 51, 87, 80, 62],
	[18, 22, 37, 56, 68, 109, 103, 77],
	[24, 35, 55, 64, 81, 104, 113, 92],
	[49, 64, 78, 87, 103, 121, 120, 101],
	[72, 92, 95, 98, 112, 100, 103, 99]
])
class compression(Enum):
	Uncompressed = 0
	RLE = 1

def main():
	def quality_type(x):
		x = int(x)
		if x < 1 or x > 100:
			raise argparse.ArgumentTypeError('Quality must be between 1 and 100 (both inclusive)')
		return x
	# set up parser
	parser = argparse.ArgumentParser(description = 'DCT Compression')
	subparsers = parser.add_subparsers()
	# set up parser for compression
	parser_compress = subparsers.add_parser('compress', help = 'Compress an image')
	parser_compress.set_defaults(func = compress)
	parser_compress.add_argument('--quality', '-q', help = 'Image quality (Default: 50, Range: 1 - 100)', type = quality_type, default = 50)
	parser_compress.add_argument('input_file', help = 'Input image')
	parser_compress.add_argument('output_file', help = 'Output file')
	# set up parser for decompression
	parser_decompress = subparsers.add_parser('decompress', help = 'Decompress an image')
	parser_decompress.set_defaults(func = decompress)
	parser_decompress.add_argument('--display', '-d', help = 'Display decompressed image', action = 'store_true')
	parser_decompress.add_argument('--output', '-o', help = 'Output file')
	parser_decompress.add_argument('input_file', help = 'Input image')

	# parse args
	args = parser.parse_args()
	if not 'func' in args:
		parser.print_usage()
		return
	args.func(args)

def compress(args):
	# load input image
	image = skimage.img_as_float(skimage.io.imread(args.input_file))
	quality = args.quality
	# write quality and image size to output file
	binary_data = struct.pack('>BIIB', quality, image.shape[0], image.shape[1], image.shape[2])
	print('Input size: ' + str(image.shape))
	blocks = np.ceil(np.array(image.shape[0:2]) / BLOCKSIZE).astype(np.uint8)
	print('Block count: ' + str(blocks))
	quant = get_quantization_matrix(quality)
	print('Quantization matrix: ' + str(quant))

	# make sure that image size is integer multiple of BLOCKSIZE
	image.resize((blocks[0] * BLOCKSIZE, blocks[1] * BLOCKSIZE, image.shape[2]))

	for x in range(blocks[0]):
		for y in range(blocks[1]):
			# calculate block indices
			bxs = x * BLOCKSIZE
			bxe = bxs + BLOCKSIZE
			bys = y * BLOCKSIZE
			bye = bys + BLOCKSIZE
			# get current block from image (BLOCKSIZE*BLOCKSIZE*channels array)
			block = image[bxs:bxe,bys:bye]
			# move last axis to front (color channels to front, keep x/y in order)
			block = np.moveaxis(block, -1, 0)
			# compress color channels separately
			for i in range(block.shape[0]):
				compressed = compress_block(quant_reshape_block(block[i], quant))
				comp_binary = write_compressed_binary(compressed)
				binary_data += comp_binary
	# write binary data to output file
	with open(args.output_file, 'wb') as f:
		f.write(binary_data)

def get_quantization_matrix(quality):
	if quality < 50:
		scale = 5000 / quality
	else:
		scale = 200 - 2 * quality
	table = np.around((QUANTIZATION_TABLE * scale + 50) / 100)
	# make sure that no zeros are in array
	table[table == 0] = 1
	return table

def quant_reshape_block(block, quant):
	# dct
	freq = fftpack.dctn(block - 0.5, type = 2, norm = 'ortho')
	# quantization
	freq = np.around((freq * 50) / quant).astype(np.int8)
	# reshape into 1D array
	data = np.empty((BLOCKSIZE * BLOCKSIZE), dtype = freq.dtype)
	x, y = 0, 0
	direction = 1
	for i in range(len(data)):
		data[i] = freq[x,y]
		if direction == 1 and (y == 0 or x == BLOCKSIZE - 1):
			direction = -1
			if x == BLOCKSIZE - 1:
				y += 1
			else:
				x += 1
		elif direction == -1 and (x == 0 or y == BLOCKSIZE - 1):
			direction = 1
			if y == BLOCKSIZE - 1:
				x += 1
			else:
				y += 1
		else:
			x += direction
			y -= direction
	return data

def compress_block(data):
	# compress with RLE
	rle = np.array([(len(list(group)), name) for name, group in itertools.groupby(data)])
	# check that length of RLE is less than uncompressed data
	if (len(rle) * 2) > len(data):
		return (compression.Uncompressed, data)
	else:
		return (compression.RLE, rle)

def write_compressed_binary(data):
	comp = data[0]
	data = data[1]
	binary = struct.pack('>BB', comp.value, len(data))
	for i in range(len(data)):
		if comp == compression.Uncompressed:
			binary += struct.pack('>b', data[i])
		elif comp == compression.RLE:
			binary += struct.pack('>Bb', data[i][0], data[i][1])
	return binary


def read_and_unpack(fmt, file):
	length = struct.calcsize(fmt)
	return struct.unpack(fmt, file.read(length))

if __name__ == '__main__':
	main()
