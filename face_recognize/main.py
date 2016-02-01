#coding:utf-8

'''
Created on 2015年7月9日

@author: Administrator
'''

import pylab as pl

from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC
from multilayer_perceptron import MultilayerPerceptronClassifier

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    pl.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    pl.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        pl.subplot(n_row, n_col, i + 1)
        pl.imshow(images[i].reshape((h, w)), cmap='gray')
        pl.title(titles[i], size=12)
        pl.xticks(())
        pl.yticks(())

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

if __name__ == '__main__':
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    n_samples, h, w = lfw_people.images.shape  # for ploting
    
    X = lfw_people.data
    n_features = X.shape[1]
    
    y = lfw_people.target
    target_names = lfw_people.target_names
    n_classes = target_names.shape[0]
    print "Total dataset size:"
    print "n_samples: %d" % n_samples
    print "n_features: %d" % n_features
    print "n_classes: %d" % n_classes
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    
    # PCA降维
    
    n_components = 150
    
    pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
    eigenfaces = pca.components_.reshape((n_components, h, w))
    
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    #param_grid = {'C': [1e3, 5e3],
              #'gamma': [0.0001, 0.0005],}
    param_grid = {'hidden_layer_sizes': [500,1000,2000,3000],
    'max_iter': [100,200,500,1000,2000],}
    #clf = GridSearchCV(SVC(kernel='rbf', class_weight='auto'), param_grid, n_jobs=-1)
    #clf = clf.fit(X_train_pca, y_train)
    clf = GridSearchCV(MultilayerPerceptronClassifier(hidden_layer_sizes=500,max_iter=100), param_grid, n_jobs=-1)
    clf.fit(X_train_pca, y_train)
    
    
    y_pred = clf.predict(X_test_pca)
    
    print classification_report(y_test, y_pred, target_names=target_names)
    print confusion_matrix(y_test, y_pred, labels=range(n_classes))
    
    #plot
    prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]
    
    plot_gallery(X_test, prediction_titles, h, w)
    
    eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
    plot_gallery(eigenfaces, eigenface_titles, h, w)

    pl.show()




