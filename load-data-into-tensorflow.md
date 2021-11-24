# example - load data into TensorFlow

## tensorflow datasets

TensorFlow comes out of the box with a lot of datasets.

**set custom  splits**

```
The datasets are already splitted in training and teststes, but of course you can setup a different split.

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
import numpy as np

def augmentimages(image, label):
  image = tf.cast(image, tf.float32)
  image = (image/255)
  image = tf.image.resize(image,(300,300))
  return image, label

count_data = tfds.load('cats_vs_dogs',split='train',as_supervised=True)
train_data = tfds.load('cats_vs_dogs', split='train[:80%]', as_supervised=True)
validation_data = tfds.load('cats_vs_dogs', split='train[80%:90%]', as_supervised=True)
test_data = tfds.load('cats_vs_dogs', split='train[-10%:]', as_supervised=True)
```

## TFRecords

The sets are versioned and only downloaded once, then they are cached.

## extrcat data from zip files

```
import os
import zipfile

local_zip = '/tmp/dogs-and-cats.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/dogs-and-cats')
zip_ref.close()
# Directory with our training horse pictures
train_dogs_dir = os.path.join('/tmp/dogs-and-cats/train/dogs')
# Directory with our training human pictures
train_cats_dir = os.path.join('/tmp/dogs-and-cats/train/cats')
```

## ImageDataGenerator

With the tf.keras.preprocessing.image.ImageDataGenerator  you can generate images or you can import them from a archive, have tem autoatically labeled, rotated, zoomed and so on.

Lets assume you have files in a directory structure like this:

```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```

The this command will import the data labeled according to the folder structure:

```
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

training_data = datagen.flow_from_directory('data/train/', class_mode='binary', batch_size=64)
validation_data = datagen.flow_from_directory('data/validation/', class_mode='binary', batch_size=64)
testing_data = datagen.flow_from_directory('data/test/', class_mode='binary', batch_size=64)
```

see https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator for more details.

## load from json

```
import json

inputs=[]
labels=[]

with open("/tmp/some-with-house-sizes-and-prizes.json", 'r') as f:
    datastore = json.load(f)
    for item in datastore:
      labels.append(item['price'])
      inputs.append(item['size'])

print(len(labels))
print(len(inputs))      
```

## load data from csv

```
import csv

inputs=[]
labels=[]

with open('/tmp/sample-key-value-pairs.csv', encoding='UTF-8') as csvfile:
  reader = csv.reader(csvfile, delimiter=",")
  for row in reader:
    labels.append(int(row[0]))
    inputs = row[1].lower()

print(len(labels))
print(len(inputs))
```

## tokenize with out of vocabulatory tokens

For language processing it often makes sende to convert words (sometime even phrases) into numbers. Since you don't know all existing words you might have to define a token for all unknown words.
You might want to replace symbols or remove stop words.

It could look like this:

```
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import numpy as np

imdb_sentences = []
train_data = tfds.as_numpy(tfds.load('imdb_reviews', split="train"))
for item in train_data:
    imdb_sentences.append(str(item['text']))
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000)
tokenizer.fit_on_texts(imdb_sentences)
sequences = tokenizer.texts_to_sequences(imdb_sentences)
print(tokenizer.word_index)
print(sequences[123])

from bs4 import BeautifulSoup
import string

stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
             "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do",
             "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
             "he", "hed", "hes", "her", "here", "heres", "hers", "herself", "him", "himself", "his", "how",
             "hows", "i", "id", "ill", "im", "ive", "if", "in", "into", "is", "it", "its", "itself",
             "lets", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought",
             "our", "ours", "ourselves", "out", "over", "own", "same", "she", "shed", "shell", "shes", "should",
             "so", "some", "such", "than", "that", "thats", "the", "their", "theirs", "them", "themselves", "then",
             "there", "theres", "these", "they", "theyd", "theyll", "theyre", "theyve", "this", "those", "through",
             "to", "too", "under", "until", "up", "very", "was", "we", "wed", "well", "were", "weve", "were",
             "what", "whats", "when", "whens", "where", "wheres", "which", "while", "who", "whos", "whom", "why",
             "whys", "with", "would", "you", "youd", "youll", "youre", "youve", "your", "yours", "yourself",
             "yourselves"]

table = str.maketrans('', '', string.punctuation)

imdb_sentences = []
train_data = tfds.as_numpy(tfds.load('imdb_reviews', split="train"))
for item in train_data:
    sentence = str(item['text'].decode('UTF-8').lower())
    sentence = sentence.replace(",", " , ")
    sentence = sentence.replace(".", " . ")
    sentence = sentence.replace("-", " - ")
    sentence = sentence.replace("/", " / ")
    soup = BeautifulSoup(sentence)
    sentence = soup.get_text()
    words = sentence.split()
    filtered_sentence = ""
    for word in words:
        word = word.translate(table)
        if word not in stopwords:
            filtered_sentence = filtered_sentence + word + " "
    imdb_sentences.append(filtered_sentence)

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=25000)
tokenizer.fit_on_texts(imdb_sentences)
sequences = tokenizer.texts_to_sequences(imdb_sentences)
print(tokenizer.word_index)
```
