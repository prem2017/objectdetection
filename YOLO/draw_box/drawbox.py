#!/usr/bin/python

"""Object-box generator module 

	Credit-socurce: https://github.com/genomexyz/detect_object.git
	Further adapted for this project
"""

import os
import argparse

import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
import os
import pickle


import pdb



# Dumps DL models to pickle file
def dump_pyobj(pyobj, filepath):
	# filepath = data_base_path() + filepath
	fopen = open(filepath, 'wb')
	pickle.dump(pyobj, fopen)
	fopen.close()


# Loads python object from pickle file
def load_pyobj(filepath):
	# filepath = data_base_path() + filepath
	fopen = open(filepath, 'rb')
	obj = pickle.load(fopen)
	fopen.close()
	return obj


class BoundingBox(tk.Tk):
	"""This class helps in generating bounding box around object in given image and is currently adapted for two classes only but can be extended for any number of classes.
		After running the module use mouse to bound an object on image and repeat this process until all the object are bound then
			* press key <s> to save all the boxes
			* press key <d> to delete if there was any mistake. Please note that for now user has to redraw on all of the object in 
			  currnt image if key <d> was used.
			* Use left and right key to navigate and skip if no object is present in the image. 	   

		Parameters:
		-----------
			canvas_size (tuple): width and height of the canvas on which image will be drawn
			dir_img (str): fullpath/relative-path to directory where all the images are saved
			datasave_csv (str): filename.csv for storing the (image, x1, y1, x2, y2, label, width, height) for each box
			datasave_pk (str): filename.pk for storing dictionary of {key (image): val ([[x1,y1,x2,y2,label,width,height]])}    

		Returns:
		--------
			Saves the box information in two forms- one with csv and other with key, value pair where value is list of list 
			since an image can have more than one object in the image. 	

		# TODO: allow support for multi class with module launch argument or read from some text_file 
	"""


	def __init__(self, canvas_size, dir_img, datasave_csv, datasave_pk):
		tk.Tk.__init__(self)
		self.x = self.y = 0
		self.width_canvas, self.height_canvas = canvas_size
		self.dirimg = dir_img # data
		self.datasave = datasave_csv # 'agri.csv' # 'box.csv'
		self.datasave_pk = datasave_pk # 'agri_dict.pk'

		
		self.canvas = tk.Canvas(self, width=self.width_canvas, height=self.height_canvas, cursor="cross")


		self.canvas.pack(side="top", fill="both", expand=True)
		self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
		self.canvas.bind("<ButtonPress-1>", self.on_button_press)
		self.canvas.bind("<B1-Motion>", self.on_move_press)
		self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

		self.canvas.focus_set()
		self.canvas.bind("<Left>", self.previmg)
		self.canvas.bind("<Right>", self.nextimg)
		self.canvas.bind("s", self.saveboxcord)
		self.canvas.bind("d", self.resetbox)


		#add label of numbering
		self.numbering = tk.Label(self, text='0')
		self.numbering.pack()

		#open data save
		self.allimg = sorted([ fname for fname in os.listdir(self.dirimg) if '.jpg' in fname.lower() or '.png' in fname.lower()])
		self.allimg = set(self.allimg)

		# pdb.set_trace()
		if os.path.exists(self.datasave_pk):
			completed_files = set(load_pyobj(self.datasave_pk).keys())
			self.allimg = sorted(list(self.allimg - completed_files))

		# print(self.allimg)

		 # sorted(os.listdir(self.dirimg))

		self.imgptr = 0

		self.boxdata = None

		self.allcord = []

		self.allrect = []
		self.rect = None

		self.start_x = None
		self.start_y = None
		self.end_x = None
		self.end_y = None
		self.totwidth = None
		self.totheight = None


		self.moved = 0
		self._draw_image()


	def _draw_image(self):
		self.im = Image.open(os.path.join(self.dirimg, self.allimg[self.imgptr]))
		self.tk_im = ImageTk.PhotoImage(self.im)
		self.canvas.create_image(0,0,anchor="nw",image=self.tk_im)
		self.numbering.configure(text=self.allimg[self.imgptr]) # str(self.imgptr))


	def saveboxcord(self, event):
		self.boxdata = open(self.datasave, 'a+')

		objbox_dict = {}
		if os.path.exists(self.datasave_pk):
			objbox_dict = load_pyobj(self.datasave_pk)
		
		key = self.allimg[self.imgptr]
		objbox_dict[key] = []


		for row in self.allcord:
			objbox_dict[key].append(row)
			trow = key
			for ele in row:
				trow += (',' + str(ele))
			trow += '\n'
			self.boxdata.write(trow)
		dump_pyobj(objbox_dict, self.datasave_pk)
		print('[Saved] for #Images = ', len(objbox_dict.keys()))

		for i in range(len(self.allrect)):
			self.canvas.delete(self.allrect[i])

		print('here is the problem')
		del self.allcord[:]
		del self.allrect[:]
		self.boxdata.close()
		self.numbering.configure(text="box saved")

	def resetbox(self, event):
		for i in range(len(self.allrect)):
			self.canvas.delete(self.allrect[i])
		del self.allcord[:]
		del self.allrect[:]
		self.numbering.configure(text="box reset")

	def nextimg(self, event):

		for i in range(len(self.allrect)):
			self.canvas.delete(self.allrect[i])
		del self.allcord[:]
		del self.allrect[:]

		self.canvas.delete("all")

		self.imgptr += 1
		if self.imgptr > len(self.allimg)-1:
			self.imgptr = 0
		self._draw_image()
		self.numbering.configure(text=self.allimg[self.imgptr]) # str(self.imgptr))
		# self.numbering.configure(text=str(self.imgptr))

	def previmg(self, event):
		for i in range(len(self.allrect)):
			self.canvas.delete(self.allrect[i])
		del self.allcord[:]
		del self.allrect[:]
		self.canvas.delete("all")

		self.imgptr -= 1
		if self.imgptr < 0:
			self.imgptr = len(self.allimg)-1
		self._draw_image()
		self.numbering.configure(text=self.allimg[self.imgptr]) # str(self.imgptr))


	def on_button_press(self, event):
		# save mouse drag start position
		self.start_x = event.x
		self.start_y = event.y

		# create rectangle if not yet exist
		#if not self.rect:
		self.rect = self.canvas.create_rectangle(self.x, self.y, 1, 1, outline='red', width=1)

	def on_move_press(self, event):
		curX, curY = (event.x, event.y)

		if curX > self.width_canvas:
			curX = self.width_canvas
		if curY > self.height_canvas:
			curY = self.height_canvas

		self.end_x = curX
		self.end_y = curY

		# expand rectangle as you drag the mouse
		self.canvas.coords(self.rect, self.start_x, self.start_y, curX, curY)

		margin = 15
		if self.end_x > self.start_x + margin and self.end_y > self.start_y + margin and self.end_x < self.im.width + margin and self.end_y < self.im.height + margin:
			self.moved = 1
		else:
			self.allrect.append(self.rect)




	def on_button_release(self, event):
		if not self.moved:
			print('[Moved] = ', self.moved)
			return
		else:
			print('[Moved] = ', self.moved)
			self.moved = 0  

		events =  ['char', 'delta', 'height', 'keycode', 'keysym', 'keysym_num', 'num', 'send_event', 'serial', 'state', 'time', 'type', 'widget', 'width', 'x', 'x_root', 'y', 'y_root']
		# print('\n\n[Event] ', [getattr(event, attr)() for attr in events])
		self.imarray = np.array(self.im)
		self.totwidth = self.im.width
		self.totheight = self.im.height
		print('\n\n', self.im.filename, '\n', '  ', self.start_x, self.start_y, self.end_x, self.end_y, self.totwidth, self.totheight)

		def selected():
			# [Note] {Beet: 0, Thistle: 1)
			print('Beet (value) = ', beet.get(), 'Thistle (value) = ', thistle.get())

			if thistle.get():
				self.lable = thistle.get()
			else:
				self.lable = 0

			topframe.destroy()
			popup.destroy()

			print(' self.allcord = ',  self.allcord)
			if self.allcord:
				print(' self.allcord = ',  self.allcord)
				self.allcord[-1][-3] = self.lable # TODO is hack find alternative way


		popup = tk.Toplevel(self, background='gray20')
		popup.wm_title("Label")
		topframe = tk.Frame(popup, background='gray20')
		topframe.grid(column=0, row=0)

		# NOTE: make changes here to add more classes
		beet = tk.IntVar() # ON (Thistle)/OFF (Beet)
		tk.Checkbutton(topframe, text="Beet", variable=beet, offvalue = 0, onvalue = 1, command=selected).grid(row=0,sticky='w')
		thistle = tk.IntVar()
		tk.Checkbutton(topframe, text="Thistle", variable=thistle, offvalue = 0, onvalue = 1, command=selected).grid(row=1,sticky='w')
		
		# c.pack(side="right", fill="x", anchor='nw')
		self.lable = thistle.get()



		current_coordinate = [min(self.totwidth, self.start_x), min(self.totheight, self.start_y), min(self.totwidth, self.end_x), min(self.totheight, self.end_y), self.lable, self.totwidth, self.totheight]

		err_mg = 15
		if self.allcord: 
			if self.allcord[-1] != current_coordinate and self.start_x <= self.totwidth + err_mg and self.start_y  <= self.totheight + err_mg and  self.end_x <= self.totwidth + err_mg and self.end_y <= self.totheight + err_mg:
				print('\nself.allcord[-1] = ', self.allcord[-1], '\ncurrent_coordinate = ', current_coordinate)
				self.allcord.append(current_coordinate)
		else:
			if  self.start_x <= self.totwidth + err_mg and self.start_y  <= self.totheight + err_mg and  self.end_x <= self.totwidth + err_mg and self.end_y <= self.totheight + err_mg:
				self.allcord.append(current_coordinate)

		self.allrect.append(self.rect)
		print ('Records = ', len(self.allcord))
		print('#Rect = ', self.rect)

		self.numbering.configure(text=self.allimg[self.imgptr] + '(#{})'.format(len(self.allcord))) # str(self.imgptr))


def get_argument_parser(canvas_size=(500, 500), dir_img='data', datasave='agri.csv', datasave_pk='agri_dict.pk'):
	canvas_width, canvas_height = canvas_size
	description = 'Pass arguments to set width and height, image-directory, csv-filename, pickle-filename'
	parser = argparse.ArgumentParser(description=description)

	parser.add_argument('-cw', '--canvas_width', type=int, default=canvas_width,
						help='Give the width of canvas on which image will be displayed and should be at least 100px more than max width from all of the images', required=False)

	parser.add_argument('-ch', '--canvas_height', type=int, default=canvas_height,
						help='Give the height of canvas on which image will be displayed and should be at least 100px more than max height from all of the images', required=False)


	parser.add_argument('-d', '--dir_img', type=str, default=dir_img, 
						help='Give fullpath to directory where images are located or relative from current dir', required=False)

	parser.add_argument('-s', '--save_csv_file', type=str, default=datasave,
						help='GIve filename where object box data will be saved', required=False)

	parser.add_argument('-p', '--pk_file', type=str, default=datasave_pk,
						help='Give pickle filename where object-box info is stored in dictionary with key image name', required=False)

	return parser






if __name__ == "__main__":
	parser = get_argument_parser()
	arg_parser = parser.parse_args()
	canvas_width = arg_parser.canvas_width
	canvas_height = arg_parser.canvas_height 

	dir_img = arg_parser.dir_img 
	datasave_csv = arg_parser.save_csv_file 
	datasave_pk = arg_parser.pk_file

	canvas_size = canvas_width, canvas_height
	print(f'canvas_size={canvas_size}, dir_img={dir_img}, datasave_csv={datasave_csv}, datasave_pk={datasave_pk}')



	draw = BoundingBox(canvas_size=canvas_size, dir_img=dir_img, datasave_csv=datasave_csv, datasave_pk=datasave_pk)
	draw.mainloop()










