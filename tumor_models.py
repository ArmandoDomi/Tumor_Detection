# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 13:20:13 2020

@author: Armando
"""



import os

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from sklearn.decomposition import PCA
from sklearn import svm
from sklearn import linear_model as lm
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

from sklearn.metrics.pairwise import chi2_kernel,additive_chi2_kernel
import itertools
from sklearn.metrics import confusion_matrix,plot_confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report
from skimage.feature import hog
from skimage.color import rgb2grey


#plot roc curve
def plot_roc_curve(fpr, tpr):
    '''
    A method that plots the roc curve

    Parameters
    ----------
    fpr : float
        False Positive Rate
    tpr : float
        True Positive Rate
    '''
    plt.figure(10)
    plt.plot(fpr, tpr, color='orange',label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()



def my_plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    
    Parameters
    ----------
    cm : array , int
        confusion matrix
    classes : list , str
        A list with the name of classes
    title : str 
        The title of the plot
    """
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        #print('Confusion matrix, without normalization')

    #print(cm)

        thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



#logistic Regression k-folds
def printing_Kfold_scoresLR(features,targets):

    '''
    A method that's find the best parameters for a LogisticRegression model by using  5-fold cross validation.
    The metrics to find those parameters are Recall and Accuracy.

    Parameters
    ----------
    features : array,float
        the train set
    y_test : array,float
        the targets of the train set

    Returns
    -------
    dictionary
    A dictionary with the best parameters of the model. Also the best accuracy and recall
    '''

    n_folds=5

    kf = KFold(n_splits=n_folds)

    Cs=[0.001,0.01,0.1,1,10,100]
    #variables for best param
    best_Param={}
    best_score=-1

    for C in Cs:
        print("C:{} \n".format(C))

        accuracy = precision = recall = sensitivity = specificity=fmeasure = 0

        for train_index, test_index in kf.split(features):

            X_train, X_test, y_train, y_test = features[train_index], features[test_index], targets[train_index], targets[test_index]
            model=lm.LogisticRegression(C=C,penalty='l1',n_jobs=-1,solver='saga')
            model.fit(X_train,y_train)

            y_pred=model.predict(X_test)

            tn, fp, fn, tp =confusion_matrix(y_test,y_pred).ravel()

            accuracy=((tn+tp)/(tn+tp+fn+fp))+accuracy
            precision=(tp/(tp+fp))+precision
            recall=(tp/(tp+fn))+recall
            fmeasure=(((precision*recall*2)/((precision+recall))))+fmeasure
            sensitivity=(tp/(tp+fn))+sensitivity
            specificity=(tn/(tn+fp))+specificity


        print("Mean Accuracy :",accuracy/n_folds)
        print("Mean Precision :",precision/n_folds)
        print("Mean recall :",recall/n_folds)
        print("Mean Fmeasure :",fmeasure/n_folds)
        print("Mean Sensitivity :",sensitivity/n_folds)
        print("Mean Specificity :",specificity/n_folds)
        print()

        #find the best recall
        if ((recall/n_folds) +(accuracy/n_folds))/2 > best_score :
                best_score=((recall/n_folds) +(accuracy/n_folds))/2
                best_Param={'C':C,'recall':recall/n_folds,'accuracy':accuracy/n_folds}

    print("Best param is:{}".format(best_Param))
    return best_Param


#support vector machine k-folds
def printing_Kfold_scoresSVM(features,targets,kernel):

    '''
    A method that's find the best parameters for a SVM model by using  5-fold cross validation.
    The metrics to find those parameters are Recall and Accuracy.

    Parameters
    ----------
    features : array,float
        the train set
    y_test : array,float
        the targets of the train set
    kernel: str
        the kernel of the svm model.It must be linear or fbf


    Returns
    -------
    dictionary
    A dictionary with the best parameters of the model. Also the best accuracy and recall
    '''

    n_folds=5

    kf = KFold(n_splits=n_folds)

    Cs=[1, 10, 100, 1000]
    gammas=[0.1,0.01,0.001, 0.0001]
    #variables for best param
    best_Param={}
    best_score=-1

    for C in Cs:

        for gamma in gammas:

            print("C:{} and gamma:{} \n".format(C,gamma))

            accuracy = precision = recall = sensitivity = specificity=fmeasure = 0

            for train_index, test_index in kf.split(features):


                X_train, X_test, y_train, y_test = features[train_index], features[test_index], targets[train_index], targets[test_index]
                model=svm.SVC(C=C,gamma=gamma,kernel=kernel)
                model.fit(X_train,y_train)

                y_pred=model.predict(X_test)

                tn, fp, fn, tp =confusion_matrix(y_test,y_pred).ravel()

                accuracy=((tn+tp)/(tn+tp+fn+fp))+accuracy
                precision=(tp/(tp+fp))+precision
                recall=(tp/(tp+fn))+recall
                fmeasure=(((precision*recall*2)/((precision+recall))))+fmeasure
                sensitivity=(tp/(tp+fn))+sensitivity
                specificity=(tn/(tn+fp))+specificity


            print("Mean Accuracy :",accuracy/n_folds)
            print("Mean Precision :",precision/n_folds)
            print("Mean recall :",recall/n_folds)
            print("Mean Fmeasure :",fmeasure/n_folds)
            print("Mean Sensitivity :",sensitivity/n_folds)
            print("Mean Specificity :",specificity/n_folds)
            print()

            #find the best recall
            if ((recall/n_folds) +(accuracy/n_folds))/2 > best_score :
                best_score=((recall/n_folds) +(accuracy/n_folds))/2
                best_Param={'C':C,'gamma':gamma,'recall':recall/n_folds,'accuracy':accuracy/n_folds}


    print("Best param is:{}".format(best_Param))
    return best_Param

#Random Forest k-folds
def printing_Kfold_scoresRF(features,targets):

    '''
    A method that's find the best parameters for a Random Forest model by using  5-fold cross validation.
    The metrics to find those parameters are Recall and Accuracy.

    Parameters
    ----------
    features : array,float
        the train set
    y_test : array,float
        the targets of the train set

    Returns
    -------
    dictionary
    A dictionary with the best parameters of the model. Also the best accuracy and recall
    '''

    n_folds=5

    kf = KFold(n_splits=n_folds)


    estimators= [50,100, 500, 1000]
    max_depths=[80, 100, 110, 120]

    best_Param={}
    best_score=-1

    for n_estimators in estimators:

        for max_depth in max_depths:

            print("n_estimators:{} and max_depth:{} \n".format(n_estimators,max_depth))

            accuracy = precision = recall = sensitivity = specificity=fmeasure = 0

            for train_index, test_index in kf.split(features):


                X_train, X_test, y_train, y_test = features[train_index], features[test_index], targets[train_index], targets[test_index]
                model=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,n_jobs=-1)
                model.fit(X_train,y_train)

                y_pred=model.predict(X_test)

                tn, fp, fn, tp =confusion_matrix(y_test,y_pred).ravel()

                accuracy=((tn+tp)/(tn+tp+fn+fp))+accuracy
                precision=(tp/(tp+fp))+precision
                recall=(tp/(tp+fn))+recall
                fmeasure=(((precision*recall*2)/((precision+recall))))+fmeasure
                sensitivity=(tp/(tp+fn))+sensitivity
                specificity=(tn/(tn+fp))+specificity


            print("Mean Accuracy :",accuracy/n_folds)
            print("Mean Precision :",precision/n_folds)
            print("Mean recall :",recall/n_folds)
            print("Mean Fmeasure :",fmeasure/n_folds)
            print("Mean Sensitivity :",sensitivity/n_folds)
            print("Mean Specificity :",specificity/n_folds)
            print()

            #find the best recall and accu
            if ((recall/n_folds) +(accuracy/n_folds))/2 > best_score :

                best_score=((recall/n_folds) +(accuracy/n_folds))/2
                best_Param={'n_estimators':n_estimators,'max_depth':max_depth,'recall':recall/n_folds,'accuracy':accuracy/n_folds}


    print("Best param is:{}".format(best_Param))
    return best_Param


def getFeatures(path):
    '''
    a method that takes as input the path of images and return a vector with
    features of the images by using as feature extractore HOG method

    Parameters
    ----------
    path : str
        the path of the images


    Returns
    -------
    list
    A list of features for each image
    '''

    features_list = []
    for (dirpath,_, files) in os.walk(path):
        for f in files:
            img = Image.open(os.path.join(dirpath, f))
            out=img.resize((250,250)) #250
            img_array=np.array(out)
            #color_features = img_array.flatten()
            grey_image = rgb2grey(img_array)
            #hog_features = hog(grey_image, block_norm='L2-Hys', pixels_per_cell=(32, 32))
            hog_features = hog(grey_image, pixels_per_cell=(32, 32))
            features_list.append(hog_features)
        return  features_list



def findBestThresholds(accuracy,recall):
    '''
    
    a method that takes as input two vectors and create a vector with te best score
    and returns the the position with the best score
    
    Parameters
    ----------
    accuracy : list,float
        a vector with the accuray of each threshold
    recall : list, float
        a vector with the recall of each threshold

    Returns
    -------
    int
    the position of the best score
    
    
    '''
    score=np.zeros(len(accuracy))

    for i in range(0,len(accuracy)):
        score[i]=(accuracy[i]+recall[i])/2
    #print(score)

    index=np.argmax(score)

    return index

def main():
    info={}
    option=-1
    stop=False
    while not stop:

        print("Choose Dataset : \n");
        print("1. Dataset with total of 200 images\n")
        print('2. Dataset with total of 253 images \n')
        print('3. Exit\n')

        option=int(input("Choose option : "))



        #we init the lists to the null list []
        feature_matrix_no=[]
        feature_matrix_yes=[]
        target_no=[]
        target_yes=[]

        # we load the images
        if option == 1:
            info["dataset"]="Dataset with total of 200 images"
            feature_matrix_no=getFeatures('./brain-tumor-images-dataset/no')
            feature_matrix_yes=getFeatures('./brain-tumor-images-dataset/yes')

        elif option ==2:
            info["dataset"]="Dataset with total of 253 images"
            feature_matrix_no=getFeatures('./brain-mri-images-for-brain-tumor-detection/no')
            feature_matrix_yes=getFeatures('./brain-mri-images-for-brain-tumor-detection/yes')

        elif option == 3:

            print("Bye\n")
            break

        else:
            print("Wrong input, please type 1 or 2 or 3\n")
            continue

        # we create the targets of the images
        target_no=np.zeros(len(feature_matrix_no))
        target_yes=np.ones(len(feature_matrix_yes))


        #finally we combine the two array features and targets
        targets=np.concatenate((target_no,target_yes))
        feature_matrix=np.array(feature_matrix_no+feature_matrix_yes)

        # Let's print the feature matrix shape
        print('Feature matrix shape is: ', feature_matrix.shape)

        #plot a frequency bar
        classes = ('Class No','Class Yes')
        y_pos = np.arange(len(classes))
        performance = [len(target_no),len(target_yes)]

        plt.bar(y_pos, performance, align='center', alpha=0.5,color=('b','red'))
        plt.xticks(y_pos, classes)
        plt.title('Tumor class histogram')
        plt.xlabel("Frequency")
        plt.show()

        print("Choose  1 for feature reduction : \n");
        print("1. Principal Component Analysis\n")
        print("2. For no change to the features\n")
        print('3. Exit\n')

        option=int(input("Choose option : "))
        final_features=[]

        if option == 1:
            info["feature reduction"]=True
            # after try and error we choose 0.9 parameter for pca
            pca=PCA(0.9)
            pca_features = pca.fit_transform(feature_matrix)
            final_features=pca_features
            print('Feature matrix shape after PCA: \n', final_features.shape)

        elif option ==2:
            info["feature reduction"]=False
            final_features=feature_matrix

        elif option == 3:

            print("Bye")
            break

        else:
            print("Wrong input, please type 1 or 2 or 3 \n")
            continue


        #split the data  70% training ,15% validation and 15% test
        
        #X_train, X_test, y_train, y_test = train_test_split(final_features,targets,test_size = 0.3)
        X_train, X_test, y_train, y_test = train_test_split(final_features,targets,test_size = 0.3, random_state = 0)
        X_val, X_test, y_val, y_test = train_test_split(X_test,y_test,test_size = 0.5, random_state = 0)

        print
        print("\nThe training set is :"+str(X_train.shape[0]))
        print("\nThe test set is :"+str(X_test.shape[0]))
        print("\nThe val set is :"+str(X_val.shape[0]))

        print("Choose a method to train your model\n")
        print("1. SVM\n")
        print("2. Linear-SVM\n")
        print("3. Additive Chi-squared kernel\n")
        print('4. LogisticRegression\n')
        print("5. RandomForest\n")
        print('6. Exit\n')

        option=int(input("Choose option : "))
        final_features=[]

        if option == 1:
            info["model"]="SVM"
            kernel="rbf"
            #we create the svm model with the best parameters
            b_param=printing_Kfold_scoresSVM(X_train,y_train,kernel)
            model= svm.SVC(C=b_param['C'],gamma=b_param['gamma'],kernel=kernel,probability=True)

        elif option ==2:
            info["model"]="Linear-SVM"
            kernel="linear"
            #we create the linear-svm model with the best parameters
            b_param=printing_Kfold_scoresSVM(X_train,y_train,kernel)
            model= svm.SVC(C=b_param['C'],gamma=b_param['gamma'],kernel=kernel,probability=True)

        elif option == 3:
            info["model"]="Additive Chi-squared kernel"
            model=svm.SVC(kernel=additive_chi2_kernel,probability=True)

        elif option == 4:
            info["model"]="LogisticRegression"
            #we create the Logistic Regression model
            b_param=printing_Kfold_scoresLR(X_train,y_train)
            model=lm.LogisticRegression(C=b_param['C'],solver='saga',penalty='l1')

        elif option == 5:
            info["model"]="RandomForest"
            #we create the random forest model with the best parameters
            b_param=printing_Kfold_scoresRF(X_train,y_train)
            model=RandomForestClassifier(n_estimators=b_param['n_estimators'],max_depth=b_param['max_depth'],n_jobs=-1)

        elif option ==6:
            print("Bye")
            break

        else:

            print("Wrong input, please type 1 or 2 or 3 or 4 or 5\n")
            continue

        # we train the model
        model.fit(X_train,y_train)
        
        #probs =model.decision_function(X_test)

        # get the prediction and probability
        y_pred=model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)

        # create the confusion_matrix
        tn, fp, fn, tp =confusion_matrix(y_val,y_pred).ravel()
        #  The list of thresholds
        thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]


        plt.figure(figsize=(10,10))

        accuracy=np.zeros(9)
        precision=np.zeros(9)
        recall=np.zeros(9)
        sensitivity=np.zeros(9)
        specificity=np.zeros(9)
        fmeasure = np.zeros(9)

        #now let's try to find the most suitable threshold
        j = 1
        for i in thresholds:
            y_val_predictions_high_recall = y_pred_proba[:,1] > i

            plt.subplot(3,3,j)


            # Compute confusion matrix
            cnf_matrix = confusion_matrix(y_val,y_val_predictions_high_recall)
            np.set_printoptions(precision=2)

            print('Threshold >= %s'%i)



            tn, fp, fn, tp =confusion_matrix(y_val,y_val_predictions_high_recall).ravel()

            print("tn:{} fp:{} fn:{} tp:{}".format(tn,fp,fn,tp))

            
            accuracy[j-1]=((tn+tp)/(tn+tp+fn+fp))
            precision[j-1]=(tp/(tp+fp))
            recall[j-1]=(tp/(tp+fn))
            fmeasure[j-1]=((precision[j-1]*recall[j-1]*2)/((precision[j-1]+recall[j-1])))
            sensitivity[j-1]=(tp/(tp+fn))
            specificity[j-1]=(tn/(tn+fp))

            print()
            print("Val set")
            print("Accuracy :",accuracy[j-1])
            print("Precision :",precision[j-1])
            print("recall :",recall[j-1])
            print("Fmeasure :",fmeasure[j-1])
            print("Sensitivity :",sensitivity[j-1])
            print("Specificity :",specificity[j-1])
            print()

            j = j+1
            # Plot non-normalized confusion matrix
            class_names = ['No','Yes']
            my_plot_confusion_matrix(cnf_matrix
                                  , classes=class_names
                                  , title='Threshold >= %s'%i)

        index=findBestThresholds(accuracy,recall)
        #print(index)
        
        
        # now let's test the model at the test set
        y_pred=model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        tn, fp, fn, tp =confusion_matrix(y_test,y_pred).ravel()
        y_test_predictions_high_recall = y_pred_proba[:,1] > thresholds[index]

        tn, fp, fn, tp  = confusion_matrix(y_test,y_test_predictions_high_recall).ravel()

        accuracy = precision = recall = sensitivity = specificity = fmeasure = 0

        accuracy=((tn+tp)/(tn+tp+fn+fp))
        precision=(tp/(tp+fp))
        recall=(tp/(tp+fn))
        fmeasure=(((precision*recall*2)/((precision+recall))))
        sensitivity=(tp/(tp+fn))
        specificity=(tn/(tn+fp))
        
        print("{}\n".format(info))
        print("**** Test set ****")
        print("tn:{} fp:{} fn:{} tp:{}".format(tn,fp,fn,tp))
        print("Accuracy :",accuracy)
        print("Precision :",precision)
        print("recall :",recall)
        print("Fmeasure :",fmeasure)
        print("Sensitivity :",sensitivity)
        print("Specificity :",specificity)
        print()
        
        
        #now draw roc curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        plot_roc_curve(fpr, tpr)


if __name__ == "__main__":
    """ Executed only when the file is run as a script. """
    main()