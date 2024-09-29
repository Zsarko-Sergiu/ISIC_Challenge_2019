#Here I'll define some methods used to filter the dataset
import tensorflow as tf
import random

def filter_dataset(ds_train):
    def filter_choice(line):
        split_line=tf.strings.split(line,",")
        name=split_line[0]
        label=''
        for i in range(1,len(split_line)):
            if split_line[i]=='1.0':
                #if its 1.0 then it will be the label
                label=split_line[i]
        #return true if name is not empty and we have a label for our data
        return True if name!=' ' and label!='' else False
    ds_train = ds_train.filter(filter_choice)
    return ds_train

def build_dataset(ds_train):
    #filter dataset
    ds_train = filter_dataset(ds_train)
    diseases=['MEL','NV','BCC','AK','BKL','DF','VASC','SCC','UNK']
    dataset=[]
    for line in ds_train:
        line=tf.strings.split(line,",")
        name=line[0]
        label=''
        type_disease=''
        for i in range(1,len(line)):
            if line[i]=='1.0':
                type_disease=diseases[i-1]
                label=str(i)
                break
        data=name+','+label+','+type_disease
        dataset.append(data)
    return dataset


def convert_to_tensor(data):
    data=tf.data.Dataset.from_tensor_slices(data)
    def convert(data_line):
        data_line=tf.strings.split(data_line,",")
        name=data_line[0]
        label=tf.strings.to_number(data_line[1],tf.float32)
        return name,label
    data=data.map(convert)
    return data

def augment(image):

    if random.random() < 0.5: #flip left right
        image=tf.image.flip_left_right(image)

    if random.random() < 0.5:  # flip up down
        image = tf.image.flip_up_down(image)

    if random.random() < 0.5: #increase brightness
        value=random.randint(0,50)
        image=tf.image.adjust_brightness(image,value)

    if random.random() < 0.5: #greyscale
        image=tf.image.rgb_to_grayscale(image)
        image=tf.image.grayscale_to_rgb(image)

    return image
#
def build_train_set(data):

    def load_image(name,label):
        #get the image from the data folder using its name
        path=tf.strings.join(["ISIC_2019_Training_Input/",name,".jpg"])

        image=tf.io.read_file(path)
        image=tf.io.decode_jpeg(image,channels=3)
        image=tf.image.resize(image,[128,128])
        image=image/255.
        image=augment(image)
        return image,label

    autotune = tf.data.AUTOTUNE
    buffer=1003
    data=data.map(load_image,num_parallel_calls=autotune) #map
    data=data.shuffle(buffer_size=buffer)
    data=data.batch(32)
    data=data.prefetch(buffer_size=autotune)
    return data


