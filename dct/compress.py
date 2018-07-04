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
	if (len(image.shape) == 2): # grayscale image with single channel
		image = image.reshape((image.shape[0], image.shape[1], 1))
	# write quality and image size to output file
	binary_data = struct.pack('>BIIB', quality, image.shape[0], image.shape[1], image.shape[2])
	print('Input size: ' + str(image.shape))
	blocks = np.ceil(np.array(image.shape[0:2]) / BLOCKSIZE).astype(np.uint8)
	print('Block count: ' + str(blocks))
	quant = get_quantization_matrix(quality)
	print('Quantization matrix: ' + str(quant))

	# make sure that image size is integer multiple of BLOCKSIZE
	image = np.resize(image, (blocks[0] * BLOCKSIZE, blocks[1] * BLOCKSIZE, image.shape[2]))

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
				# dct, compress, write binary
				quantized = quantize_block(block[i], quant)
				reshaped = reshape_block(quantized)
				compressed = compress_block(reshaped)
				binary_data += write_compressed_binary(compressed)
	# write binary data to output file
	with open(args.output_file, 'wb') as f:
		f.write(binary_data)

def decompress(args):
	with open(args.input_file, 'rb') as file:
		quality, x, y, c = read_and_unpack('>BIIB', file)
		shape = (x, y, c)
		print('Image size: ', shape)
		blocks = np.ceil(np.array(shape[0:2]) / BLOCKSIZE).astype(np.uint8)
		padded_shape = (blocks[0] * BLOCKSIZE, blocks[1] * BLOCKSIZE, c)
		print('Block count:', blocks)
		quant = get_quantization_matrix(quality)

		# create array for final image (including block padding)
		image = np.empty(padded_shape, dtype = np.float)

		# iterate through blocks
		for x in range(blocks[0]):
			for y in range(blocks[1]):
				# calculate block indices
				bxs = x * BLOCKSIZE
				bxe = bxs + BLOCKSIZE
				bys = y * BLOCKSIZE
				bye = bys + BLOCKSIZE
				# create array for block (channel*BLOCKSIZE*BLOCKSIZE)
				block = np.empty((shape[2], BLOCKSIZE, BLOCKSIZE), dtype = image.dtype)
				# iterate through color channels
				for c in range(shape[2]):
					# read binary, decompress, reverse dct
					compressed = read_compressed_binary(file)
					reshaped = decompress_block(compressed)
					quantized = reverse_reshape_block(reshaped)
					block[c] = dequantize_block(quantized, quant)
				# reorder block dimensions to (BLOCKSIZE*BLOCKSIZE*channel)
				block = np.moveaxis(block, 0, -1)
				# write current block to image
				image[bxs:bxe,bys:bye] = block
	image = image[0:shape[0],0:shape[1]]
	image = np.squeeze(image)
	if args.display == True:
		skimage.io.imshow(image)
		skimage.io.show()
	if args.output != None:
		skimage.io.imsave(args.output, image)

def get_quantization_matrix(quality):
	if quality < 50:
		scale = 5000 / quality
	else:
		scale = 200 - 2 * quality
	table = np.around((QUANTIZATION_TABLE * scale + 50) / 100)
	# make sure that no zeros are in array
	table[table == 0] = 1
	return table

def quantize_block(block, quant):
	# dct
	freq = fftpack.dctn(block - 0.5, type = 2, norm = 'ortho')
	# quantization
	return np.around((freq * 50) / quant).astype(np.int8)

def dequantize_block(block, quant):
	# dequantization
	freq = (block.astype(np.float) * quant) / 50
	# inverse dct
	block = fftpack.idctn(freq, type = 2, norm = 'ortho') + 0.5
	# clip to [0..1]
	return np.clip(block, 0, 1)

def zigzag_coords(x, y, direction):
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
	return x, y, direction

def reshape_block(block):
	# reshape into 1D array
	data = np.empty((BLOCKSIZE * BLOCKSIZE), dtype = block.dtype)
	x, y = 0, 0
	direction = 1
	for i in range(len(data)):
		data[i] = block[x, y]
		x, y, direction = zigzag_coords(x, y, direction)
	return data

def reverse_reshape_block(data):
	# reshape data into 2D array
	block = np.empty((BLOCKSIZE, BLOCKSIZE), dtype = data.dtype)
	x, y = 0, 0
	direction = 1
	for i in range(len(data)):
		block[x, y] = data[i]
		x, y, direction = zigzag_coords(x, y, direction)
	return block

def compress_block(data):
	# compress with RLE
	rle = np.array([(len(list(group)), name) for name, group in itertools.groupby(data)])
	# check that length of RLE is less than uncompressed data
	if (len(rle) * 2) > len(data):
		return (compression.Uncompressed, data)
	else:
		return (compression.RLE, rle)

def decompress_block(data):
	comp = data[0]
	data = data[1]
	if comp == compression.Uncompressed:
		return data
	elif comp == compression.RLE:
		# reverse RLE
		uncomp = []
		for length, value in data:
			uncomp += [value] * length
		return np.array(uncomp, dtype = data.dtype)

def write_compressed_binary(data):
	comp = data[0]
	data = data[1]
	# write compression type and block length
	binary = struct.pack('>BB', comp.value, len(data))
	for i in range(len(data)):
		if comp == compression.Uncompressed:
			# write uncompressed data
			binary += struct.pack('>b', data[i])
		elif comp == compression.RLE:
			# write RLE data
			binary += struct.pack('>Bb', data[i][0], data[i][1])
	return binary

def read_compressed_binary(file):
	# read compression type and block length
	comp, length = read_and_unpack('>BB', file)
	comp = compression(comp)

	if comp == compression.Uncompressed:
		# read uncompressed data
		data = np.empty(length, dtype = np.int8)
		for i in range(length):
			data[i], = read_and_unpack('>b', file)
	elif comp == compression.RLE:
		# read RLE data
		data = np.empty((length, 2), dtype = np.int8)
		for i in range(length):
			data[i][0], data[i][1] = read_and_unpack('>Bb', file)
	# return compression type and data
	return (comp, data)

def read_and_unpack(fmt, file):
	length = struct.calcsize(fmt)
	return struct.unpack(fmt, file.read(length))

if __name__ == '__main__':
	main()
