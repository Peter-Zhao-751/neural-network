import numpy
import matplotlib.pyplot
import PIL

def addMargins(image, x1, y1, x2, y2, color):
    x,y=image.size
    result=PIL.Image.new(image.mode, (x+x1+x2, y+y1+y2), color)
    result.paste(image, (x1, y1))
    return result
def smallestImage(image, axis="x"):
    if axis=="x":
        x,y = image.size
    else:
        y,x = image.size
    xColors=[]
    xCutRange1 = 0;
    xCutRange2 = 0;
    for i in range(0, x):
        totalColor = 0
        for j in range(0, y):
            if axis=="x":
                totalColor+=255-image.getpixel((i,j))
            else:
                totalColor+=255-image.getpixel((j,i))
        xColors.append(totalColor/y)
    for xColorIndex in range(0, len(xColors)):
        if xColors[xColorIndex]>5:
            xCutRange1=xColorIndex
            break
    for xColorIndex in reversed(range(0, len(xColors))):
        if xColors[xColorIndex]>5:
            xCutRange2=xColorIndex
            break
    return xCutRange1, xCutRange2
            
def imageToData(*a):
    imageSize = 11
    imageColor = 4
    xAdd = 0
    yAdd = 0
    if len(a)==1:
        image = PIL.Image.open(a[0])
    else:
        image = a[1]
        
    image=image.convert('L', palette=PIL.Image.ADAPTIVE)
    xCutRange1, xCutRange2 = smallestImage(image, "x")
    yCutRange1, yCutRange2 = smallestImage(image, "y")
    image=image.crop((xCutRange1, yCutRange1, xCutRange2, yCutRange2))
    if xCutRange2-xCutRange1<yCutRange2-yCutRange1:
        xAdd = round((yCutRange2-yCutRange1-(xCutRange2-xCutRange1))/2)
    else:
        xAdd = round((xCutRange2-xCutRange1-(yCutRange2-yCutRange1))/2)
    image = addMargins(image, xAdd, yAdd, xAdd, yAdd, "white")
    image=image.resize((imageSize,imageSize))

    total=[]
    for i in range(0,imageSize):
        for j in range(0, imageSize):
            pixel = image.getpixel((i,j))
            pixel = round(pixel/(256/imageColor))
            total.append(pixel/imageColor)
            image.load()[i, j]=round(pixel*256/imageColor)
    #image.show()
    return total   

def fillMatrix(x, y, url):
    image = PIL.Image.open(url)
    xSize, ySize = image.size
    matrix = numpy.empty((0,121), float)
    matrix2 = numpy.empty((0,4), float)
    print(xSize, ySize)
    for i in range(1,x+1):
        for j in range(1,y+1):
            newImage = image.copy()
            newImage = newImage.crop(((xSize/x*i)-xSize/x, (ySize/y*j)-ySize/y, xSize/x*i, ySize/y*j))
            matrix = numpy.append(matrix, numpy.array([imageToData(0,newImage)]), axis=0)
            index = [int(i) for i in list('{0:0b}'.format(i))]
            for thing in range(len(index),4):
                index.insert(0,0)
            matrix2 = numpy.append(matrix2,numpy.array([index]), axis=0)
    return matrix, matrix2
    

class layer():
    def __init__(self, numInputs, numNeurons):
        self.numNeurons = numNeurons
        self.weight=numpy.random.uniform(size=(numInputs, numNeurons))
        self.bias=numpy.random.uniform(size=(1, numNeurons))
    def activation(self, previousData):
        self.data=1/(1+numpy.exp(-(numpy.dot(previousData, self.weight)+self.bias)))
        return self.data
    def derivative(self):
        return self.data*(1-self.data)
    def optimize(self, previousData, error, learningRate=1):
        self.weight+= numpy.dot(previousData.T, error*self.derivative())*learningRate
        self.bias+= numpy.sum(error*self.derivative(), axis=0, keepdims=True)*learningRate
    def error(self, error):
        return numpy.dot(error*self.derivative(), self.weight.T)


epicImage = imageToData("/Users/peter/Desktop/download-6 2.png")
inputData, expectedDataOut=fillMatrix(10,3,"/Users/peter/Desktop/wBCHXId.png")
layer1=layer(121, 30)
layer2=layer(layer1.numNeurons, 30)
layer3=layer(layer2.numNeurons, 30)
layer4=layer(layer3.numNeurons, 30)
layer5 = layer(layer4.numNeurons, 30)
layer6 = layer(layer5.numNeurons, 30)
layer7 = layer(layer6.numNeurons, 4)

i=0
learningRate=0.2
thing=[]

while i<200000:
    
    layer1result=layer1.activation(inputData)
    layer2result=layer2.activation(layer1result)
    layer3result=layer3.activation(layer2result)
    layer4result=layer4.activation(layer3result)
    layer5result=layer5.activation(layer4result)
    layer6result=layer6.activation(layer5result)
    layer7result=layer7.activation(layer6result)
    
    layer7error=expectedDataOut-layer7result
    layer6error=layer7.error(layer7error)
    layer5error=layer6.error(layer6error)
    layer4error=layer5.error(layer5error)
    layer3error=layer4.error(layer4error)
    layer2error=layer3.error(layer3error)
    layer1error=layer2.error(layer2error)
    
    layer7.optimize(layer6result, layer7error)
    layer6.optimize(layer5result, layer6error)
    layer5.optimize(layer4result, layer5error)
    layer4.optimize(layer3result, layer4error)
    layer3.optimize(layer2result, layer3error)
    layer2.optimize(layer1result, layer2error)
    layer1.optimize(inputData, layer1error)
    
    
    thing.append(abs(numpy.sum((expectedDataOut-layer7result))))
    i+=1
print(layer7result.round(3))
matplotlib.pyplot.plot(thing)
matplotlib.pyplot.show()

output=layer7.activation(layer6.activation(layer5.activation(layer4.activation(layer3.activation(layer2.activation(layer1.activation(numpy.array([imageToData("/Users/peter/Desktop/download-6.png")]))))))))
print(output.round(3))

