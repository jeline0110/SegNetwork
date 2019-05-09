import torch
import torch.nn as nn

class Encode(nn.module):
	def __init__(self, ch1, ch2):
		super(Encode, self).__init__()
		layers = [ nn.Conv2d(ch1,ch2,kernel_size=3,stride=2,padding=1,bias=False),
			nn.BatchNorm2d(ch2),
			nn.ReLU(inplace=True),

			nn.Conv2d(ch2,ch2,kernel_size=3,padding=1,bias=False),
			nn.BatchNorm2d(ch2),
			nn.ReLU(inplace=True) ]

		self.encode = nn.Sequential(*layers)

	def forward(self,x):
		out = self.encode(x)

		return out

class Decode(nn.module):
	def __init__(self,ch1,ch2):
		super(Decode, self).__init__()
		layers = [ nn.ConvTranspose2d(ch1, ch2, 
                        kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ch2),
            nn.ReLU(inplace=True),

            nn.Conv2d(ch2, ch2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch2),
            nn.ReLU(inplace=True) ]

        self.decode = nn.Sequential(*layers)

	def forward(self,layer1,layer2):
		out = torch.cat((layer1,layer2),1)
		out = self.decode(out)

		return out

class My_Unet(nn.module):
	def __init__(self, input_channel, class_num):
		super(My_Unet, self).__init__()
		self.input_channel = input_channel
		self.class_num =  class_num
		self.conv1 = nn.Conv2d(self.input_channel,128,kernel_size=5,padding=2)
		self.encode2 = Encode(128,128)
		self.encode3 = Encode(128,128)
		self.encode4 = Encode(128,128)
		self.encode5 = Encode(128,128)
		self.conv6 = nn.Sequential( nn.Conv2d(128,128,kernel_size=4,bias=False),
			nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),)

		self.deconv7 = nn.Sequential( nn.ConvTranspose2d(128,128,kernel_size=4,bias=False),
			nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),)
		self.decode8 = Decode(256,128)
		self.decode9 = Decode(256,128)
		self.decode10 = Decode(256,128)
		self.decode11 = Decode(256,128)
        self.conv12 = nn.Conv2d(128,self.class_num,kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.ConvTranspose2d): 
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0)

        def forward(self,x):

            ly1 = self.conv1(x)

            ly2 = self.encode2(ly1)
            ly3 = self.encode3(ly2)
            ly4 = self.encode4(ly3)
            ly5 = self.encode5(ly4)

            ly6 = self.conv6(ly5)
            ly7 = self.deconv7(ly6)

            ly8 = self.decode8(ly7,ly5)
            ly9 = self.decode9(ly8,ly4)
            ly10 = self.decode10(ly9,ly3)
			ly11 = self.decode11(ly10,ly2)

			ly12 = self.conv12(ly11)
			score_map = self.sigmoid(ly12)

			return score_map
		