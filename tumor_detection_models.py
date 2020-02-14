
import os

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn import linear_model as lm

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image


from skimage.feature import hog
from skimage.color import rgb2grey




from sklearn.feature_selection import chi2, SelectKBest

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
    plt.plot(fpr, tpr, color='orange',label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


def plot_matrix(model,X_test, y_test):
    '''
    A method that plots the confusion matrix
    
    Parameters
    ----------
    model : object
        the trained model
    X_test : array,float
        the test set
    y_test : array,float
        the targets of the test set 
    
    '''
    # Plot non-normalized confusion matrix
    titles_options = [("Confusion matrix, without normalization", None)]
    class_names=["No","Yes"]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(model, X_test, y_test,
                                     display_labels=class_names,
                                     cmap=plt.cm.Blues,
                                     normalize=normalize)
        disp.ax_.set_title(title)
    
        print(title)
        print(disp.confusion_matrix)
        plt.show()


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
    
    class_weight={0:1.,1:4.}
    
    Cs=[0.001,0.001,0.01,0.1,1]
    #variables for best param
    best_Param={}
    best_score=-1

    for C in Cs:
        print("C:{} \n".format(C))
            
        accuracy = precision = recall = sensitivity = specificity=fmeasure = 0
        
        for train_index, test_index in kf.split(features):
            
            X_train, X_test, y_train, y_test = features[train_index], features[test_index], targets[train_index], targets[test_index]
            model=lm.LogisticRegression(C=C,class_weight=class_weight,penalty='l1',n_jobs=-1,solver='saga')
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
    
    class_weight={0:1.,1:4}
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
                model=svm.SVC(C=C,gamma=gamma,kernel=kernel,class_weight=class_weight)
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
    

    class_weight={0:1.,1:4.}
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
                model=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,class_weight=class_weight,n_jobs=-1)
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
            hog_features = hog(grey_image, block_norm='L2-Hys', pixels_per_cell=(32, 32))
            features_list.append(hog_features)
        return  features_list      


def main():
    option=-1
    stop=True
    while stop:
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
        
        
        accuracy = precision = recall = sensitivity = specificity=fmeasure = 0
    
    
        # we load the images
        if option == 1:
            
            feature_matrix_no=getFeatures('./brain-tumor-images-dataset/no')
            feature_matrix_yes=getFeatures('./brain-tumor-images-dataset/yes')
            
        elif option ==2:    
            
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
    
        print("Choose  1 for feature reduction or 2 for feature selection : \n");
        print("1. Principal Component Analysis\n")
        print('2. Chi-Square\n')
        print("3. For no change to the features\n")
        print('4. Exit\n')
        
        option=int(input("Choose option : "))
        final_features=[]
        
        if option == 1:
            # after try and error we choose 0.9 parameter for pca
            pca=PCA(0.9)
            pca_features = pca.fit_transform(feature_matrix)
            final_features=pca_features
            
        elif option ==2:    
            
            chi2_selector = SelectKBest(chi2, k=100)
            X_kbest = chi2_selector.fit_transform(feature_matrix, targets)
            final_features=X_kbest
            
        elif option == 3:
            
            final_features=feature_matrix
            
        elif option == 4:
            
            print("Bye")
            break
        
        else:
            print("Wrong input, please type 1 or 2 or 3 or 4\n")
            continue
        
    
        #split the data  70% training and 30% validation
        X_train, X_test, y_train, y_test = train_test_split(final_features,targets,test_size = 0.3)
        #X_train, X_test, y_train, y_test = train_test_split(final_features,targets,test_size = 0.3, random_state = 0)
        
        # We set the class_weight for class no to 1 and for the class yes to 4
        class_weight={0:1,1:4}
        
        print("Choose a method to train your model\n")
        print("1. SVM\n")
        print("2. Linear-SVM\n")
        print('3. LogisticRegression\n')
        print("4. RandomForest\n")
        print('5. Exit\n')
        
        option=int(input("Choose option : "))
        final_features=[]
        
        if option == 1:
            
            kernel="rbf"
            #we create the svm model with the best parameters
            b_param=printing_Kfold_scoresSVM(X_train,y_train,kernel)
            model= svm.SVC(C=b_param['C'],gamma=b_param['gamma'],kernel=kernel,class_weight=class_weight)
            
        elif option ==2:    
            
            kernel="linear"
            #we create the linear-svm model with the best parameters
            b_param=printing_Kfold_scoresSVM(X_train,y_train,kernel)
            model= svm.SVC(C=b_param['C'],gamma=b_param['gamma'],kernel=kernel,class_weight=class_weight)
            
        elif option == 3:
            
            b_param=printing_Kfold_scoresLR(X_train,y_train)
            #we create the Logistic Regression model
            model=lm.LogisticRegression(C=b_param['C'],solver='saga',class_weight=class_weight,penalty='l1')
            
        elif option == 4:
            
            #we create the random forest model with the best parameters
            b_param=printing_Kfold_scoresRF(X_train,y_train)
            model=RandomForestClassifier(n_estimators=b_param['n_estimators'],max_depth=b_param['max_depth'],class_weight=class_weight,n_jobs=-1)
            
        elif option == 5:
            
            print("Bye")
            break
        
        else:
            
            print("Wrong input, please type 1 or 2 or 3 or 4 or 5\n")
            continue
    
     
        model.fit(X_train,y_train)
        #probs =model.decision_function(X_test)
        
        y_pred=model.predict(X_test)
        
        tn, fp, fn, tp =confusion_matrix(y_test,y_pred).ravel()
        
        print("tn:{} fp:{} fn:{} tp:{}".format(tn,fp,fn,tp))
        
        accuracy=((tn+tp)/(tn+tp+fn+fp))+accuracy
        precision=(tp/(tp+fp))+precision
        recall=(tp/(tp+fn))+recall
        fmeasure=(((precision*recall*2)/((precision+recall))))+fmeasure
        sensitivity=(tp/(tp+fn))+sensitivity
        specificity=(tn/(tn+fp))+specificity
        
        
        print("Test set")
        print("Mean Accuracy :",accuracy)
        print("Mean Precision :",precision)
        print("Mean recall :",recall)   
        print("Mean Fmeasure :",fmeasure)
        print("Mean Sensitivity :",sensitivity)
        print("Mean Specificity :",specificity)
        
        #plot the confusion matrix
        plot_matrix(model,X_test, y_test)
        
        #now draw roc curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        plot_roc_curve(fpr, tpr)


if __name__ == "__main__":
    """ Executed only when the file is run as a script. """
    main()