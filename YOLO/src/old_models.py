# -*- coding: utf-8 -*-


"""	©Prem Prakash
	OtherCNN architecture tried are here and is for reference oly  
"""

""" # TODO: backup simpler network

				nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
				nn.BatchNorm2d(32),
				nn.LeakyReLU(0.1, inplace = True),


				nn.MaxPool2d(kernel_size=2, stride=2),


				nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
				nn.BatchNorm2d(64),
				nn.LeakyReLU(0.1, inplace = True),


				nn.MaxPool2d(kernel_size=2, stride=2),


				nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
				nn.BatchNorm2d(128),
				nn.LeakyReLU(0.1, inplace = True),


				nn.MaxPool2d(kernel_size=2, stride=2),


				nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
				nn.BatchNorm2d(256),
				nn.LeakyReLU(0.1, inplace = True),


				nn.MaxPool2d(kernel_size=2, stride=2),


				nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
				nn.BatchNorm2d(128),
				nn.LeakyReLU(0.1, inplace = True),


				nn.MaxPool2d(kernel_size=2, stride=2),


				nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
				nn.BatchNorm2d(64),
				nn.LeakyReLU(0.1, inplace = True),


				nn.MaxPool2d(kernel_size=2, stride=2),


				nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
				nn.BatchNorm2d(128),
				nn.LeakyReLU(0.1, inplace = True),
				nn.Conv2d(in_channels=128, out_channels=7, kernel_size=3, stride=1, padding=1),

"""

""" Done
nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1),
				nn.BatchNorm2d(16),
				nn.LeakyReLU(0.1, inplace = True),

				nn.MaxPool2d(kernel_size=2, stride=2),

				nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
				nn.BatchNorm2d(32),
				nn.LeakyReLU(0.1, inplace = True),
				
				nn.MaxPool2d(kernel_size=2, stride=2),

				nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
				nn.BatchNorm2d(64),
				nn.LeakyReLU(0.1, inplace = True),
				nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
				nn.BatchNorm2d(32),
				nn.LeakyReLU(0.1, inplace = True),

				nn.MaxPool2d(kernel_size=2, stride=2),

				nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
				nn.BatchNorm2d(64),
				nn.LeakyReLU(0.1, inplace = True),
				nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
				nn.BatchNorm2d(128),
				nn.LeakyReLU(0.1, inplace = True),

				nn.MaxPool2d(kernel_size=2, stride=2),

				nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
				nn.BatchNorm2d(256),
				nn.LeakyReLU(0.1, inplace = True),
				nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
				nn.BatchNorm2d(128),
				nn.LeakyReLU(0.1, inplace = True),

				nn.MaxPool2d(kernel_size=2, stride=2),


				nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
				nn.BatchNorm2d(64),
				nn.LeakyReLU(0.1, inplace = True),
				nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
				nn.BatchNorm2d(128),
				nn.LeakyReLU(0.1, inplace = True),

				nn.MaxPool2d(kernel_size=2, stride=2),

				nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
				nn.BatchNorm2d(256),
				nn.LeakyReLU(0.1, inplace = True),
				nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
				nn.BatchNorm2d(128),
				nn.LeakyReLU(0.1, inplace = True),
				nn.Conv2d(in_channels=128, out_channels=7, kernel_size=3, stride=1, padding=1),
"""

class AgriNetDeeper(nn.Module):
	"""docstring for AgriNet

	"""
	def __init__(self, in_channels=3, num_classes=2, anchors=1): # 32*filters, 32*filters
		# TODO: works only for one anchor per cell
		super(AgriNetDeeper, self).__init__()
		assert anchors == 1, 'Currently this method only support for <one> anchor per cell.'

		self.num_classes = num_classes
		self.anchors = 1

		filters = 16
		self.yolo_arch = nn.Sequential( \
				nn.Conv2d(in_channels=in_channels, out_channels=2*filters, kernel_size=3, stride=1, padding=1),
				nn.BatchNorm2d(2*filters),
				nn.LeakyReLU(0.1, inplace = True),
				nn.Conv2d(in_channels=2*filters, out_channels=2*filters, kernel_size=3, stride=1, padding=1),
				nn.BatchNorm2d(2*filters),
				nn.LeakyReLU(0.1, inplace = True),

				nn.MaxPool2d(kernel_size=2, stride=2),

				nn.Conv2d(in_channels=2*filters, out_channels=4*filters, kernel_size=3, stride=1, padding=1),
				nn.BatchNorm2d(4*filters),
				nn.LeakyReLU(0.1, inplace = True),
				nn.Conv2d(in_channels=4*filters, out_channels=4*filters, kernel_size=3, stride=1, padding=1),
				nn.BatchNorm2d(4*filters),
				nn.LeakyReLU(0.1, inplace = True),

				nn.MaxPool2d(kernel_size=2, stride=2),

				nn.Conv2d(in_channels=4*filters, out_channels=8*filters, kernel_size=3, stride=1, padding=1),
				nn.BatchNorm2d(8*filters),
				nn.LeakyReLU(0.1, inplace = True),
				nn.Conv2d(in_channels=8*filters, out_channels=4*filters, kernel_size=3, stride=1, padding=1),
				nn.BatchNorm2d(4*filters),
				nn.LeakyReLU(0.1, inplace = True),

				nn.MaxPool2d(kernel_size=2, stride=2),

				nn.Conv2d(in_channels=4*filters, out_channels=8*filters, kernel_size=3, stride=1, padding=1),
				nn.BatchNorm2d(8*filters),
				nn.LeakyReLU(0.1, inplace = True),
				nn.Conv2d(in_channels=8*filters, out_channels=16*filters, kernel_size=3, stride=1, padding=1),
				nn.BatchNorm2d(16*filters),
				nn.LeakyReLU(0.1, inplace = True),

				nn.MaxPool2d(kernel_size=2, stride=2),

				nn.Conv2d(in_channels=16*filters, out_channels=32*filters, kernel_size=3, stride=1, padding=1),
				nn.BatchNorm2d(32*filters),
				nn.LeakyReLU(0.1, inplace = True),
				nn.Conv2d(in_channels=32*filters, out_channels=16*filters, kernel_size=3, stride=1, padding=1),
				nn.BatchNorm2d(16*filters),
				nn.LeakyReLU(0.1, inplace = True),

				nn.MaxPool2d(kernel_size=2, stride=2),


				nn.Conv2d(in_channels=16*filters, out_channels=8*filters, kernel_size=3, stride=1, padding=1),
				nn.BatchNorm2d(8*filters),
				nn.LeakyReLU(0.1, inplace = True),
				nn.Conv2d(in_channels=8*filters, out_channels=16*filters, kernel_size=3, stride=1, padding=1),
				nn.BatchNorm2d(16*filters),
				nn.LeakyReLU(0.1, inplace = True),

				nn.MaxPool2d(kernel_size=2, stride=2),

				nn.Conv2d(in_channels=16*filters, out_channels=32*filters, kernel_size=3, stride=1, padding=1),
				nn.BatchNorm2d(32*filters),
				nn.LeakyReLU(0.1, inplace = True),
				nn.Conv2d(in_channels=32*filters, out_channels=16*filters, kernel_size=3, stride=1, padding=1),
				nn.BatchNorm2d(16*filters),
				nn.LeakyReLU(0.1, inplace = True),
				nn.Conv2d(in_channels=16*filters, out_channels=7, kernel_size=3, stride=1, padding=1),
				# nn.BatchNorm2d(16*filters),
			)



	def forward(self, X):
		output = self.yolo_arch(X)
		# print('[Output] shape-1 = ', output.shape) # [Output] shape-1 =  torch.Size([None, 7, 4, 4])
		output = output.permute(0, 2, 3, 1) # 0 is batch size
		# print('[Output] shape-2 = ', output.shape) # [Output] shape-2 =  torch.Size([None, 4, 4, 7])
		output = output.unsqueeze(3)
		# print('[Output] shape-3 = ', output.shape) # [Output] shape-3 =  torch.Size([None, 4, 4, 1, 7])
		return output