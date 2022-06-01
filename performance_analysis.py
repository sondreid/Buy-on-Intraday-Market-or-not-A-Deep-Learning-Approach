"""

Model evaluation script

"""

# adding Folder_2 to the system path
#sys.path.insert(0, '/models'/)

from analysis import *
import sys
#sys.path.insert(0,'/models/')
try:
    cwd = os.getcwd()
    os.chdir("models")
except:
    print("> Already in models directory")


#from run_tft import *

from difference_classifiers import * 

from sklearn.metrics import roc_curve, roc_auc_score

### ROC curve for MLP classiifer


## Get test predictions for all folds
classifier_preds_and_actuals = retrieve_forecasts("difference_probabilities/optimal_model")

benchmark_classifications = pd.read_csv("difference_probabilities/complete_run/Benchmark classifier.csv")


# Compare accuarcy
classifier_metrics(classifier_preds_and_actuals.intraday_price_difference_actuals, classifier_preds_and_actuals.intraday_price_difference_classifications)





def plot_roc(actuals, preds, filename,  title = None):
    """
    Function that draws a ROC curve based o
    Parameters:
        @ytrain: target feature training set
        @x_train: features training set
        @y_test: target feature test set
        @x_test: features test set
        @model: a fitted model
    """

    sns.set_style(sns_theme)
    sns.set_style({'font.family':'serif', 'font.serif':'Computer Modern'})
    actuals_binary = (actuals > 0).astype(int)

    fpr, tpr, tr = roc_curve(actuals_binary, preds)
    auc = roc_auc_score(actuals_binary, preds)


    
    plt.figure(num = None, figsize = (10,10), dpi = 150)
    plt.plot((1,0), (1,0), ls = "--", c = ".3", label = 'No skill benchmark')
    plt.xlabel('False positive rate',  fontsize=14)
    plt.plot(fpr, tpr, label = 'Model classifications on test data')
    plt.ylabel('True positive rate', fontsize=14)
    if title is not None: plt.title(title,  fontsize=17)
    plt.legend(fontsize=13)
    plt.savefig('../images/roc_curve/' + filename + '.png', bbox_inches='tight', edgecolor='none', facecolor="white")
    plt.show()
    
    return plt



# MLP roc


plot_roc(classifier_preds_and_actuals.intraday_price_difference_actuals, classifier_preds_and_actuals.intraday_price_difference_classifications, 'roc_curve_classifier')





# Benchmark ROC

plot_roc(benchmark_classifications.intraday_price_difference_actuals, benchmark_classifications.intraday_price_difference_classifications, 'roc_curve_benchmark')

classifier_metrics(benchmark_classifications.intraday_price_difference_actuals, benchmark_classifications.intraday_price_difference_classifications)