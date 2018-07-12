import sklearn.datasets as ds
import sklearn.svm as svm
digits = ds.load_digits()
#print(digits.data)
#digits.target -> mi restituisce i target del ts
clf = svm.SVC(gamma = 0.001, C=100.)
clf.fit(digits.data[:-1],digits.target[:-1])
predict = clf.predict(digits.data[-1:])
print(predict)
import pickle # packege for saving result on the hard drive
piclke.dumps(clf)
# it saves the result of the svm so it is possible to load the result
# found for a future reuse
#pickle.dump(obj,namefile)
# it saves the obj in the namefile but it is necessary to open the file
#first  with open("namefile.pickle","rw") mode read or write
pickle.load()
# pickle.load( open( "save.p", "rb" ) ) example how to load a model from
#a file
digits.images.shape
# it gives the shape of the digits datases
# its a triple where (a,b,c) a indicates the number of n_samples
# b and c indicates the size of the image (like 8*8 pixels)

# dataset.shape ruturn a pair (a,b) from a loaded dataset
# where a is the number of samples and
# b is the number of features

# slice array notation python
# a[start:end] # items start through end-1
# a[start:]    # items start through the rest of the array
# a[:end]      # items from the beginning through end-1
# a[:]         # a copy of the whole array
