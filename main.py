########################################################################################################################
###################################### MIUULBANK KREDİ ONAY TAHMİNİ ####################################################
########################################################################################################################

# Problem : Özellikleri belirtildiğinde kişilerin kredi alıp alamayacaklarını tahmin edebilecek bir makine öğrenmesi
# modeli geliştirilmek istenmektedir. Model geliştirilirken gerekli olan veri analizi ve özellik mühendisliği
# adımlarından sonra tahmin modeli kurulmuştur.

# Veri seti https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset/data adresinden alınmıştır.
# Veri seti 4269 gözlem ve 12 bağımsız değişkenden oluşmaktadır. Hedef değişken "loan_status" olarak belirtilmiş olup;
# "Approved" (0) kredinin onaylandığını, "Rejected" 1 ise kredinin onaylanmadığını belirtmektedir.

# loan_id : Her bir müşterinin benzersiz kimlik numarası.
# no_of_dependents : Başvuru sahibinin bakmakla yükümlü olduğu kişi sayısı.
# education : Başvuru sahibinin eğitim seviyesi, Lisansüstü ya da Lisansüstü değil.
# self_employed : Başvuru sahibinin serbest meslek sahibi olup olmadığı.
# income_annum : Başvuru sahibinin yıllık geliri.
# loan_amount : Kredi için talep edilen toplam tutar.
# loan_term : Kredinin geri ödenmesi gereken yıl cinsinden süre.
# cibil_score : Başvuru sahibinin kredi puanı.
# residential_assets_value : Başvuru sahibinin konut varlıklarının toplam değeri.
# commercial_assets_value : Başvuru sahibinin ticari varlıklarının toplam değeri.
# luxury_assets_value : Başvuru sahibinin lüks varlıklarının toplam değeri.
# bank_asset_value : Başvuru sahibinin banka varlıklarının toplam değeri.
# loan_status : Hedef değişken. Kredinin onaylanıp onaylanmadığını açıklar.

########################### GEREKLİ KÜTÜPHANELERİN KURULMASI VE IMPORT İŞLEMLERİ########################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
from joblib import load

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_validate
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

df_ = pd.read_csv("dataset/loan_approval_dataset.csv")
df = df_.copy()


############################################# GENEL RESİM ##############################################################
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("#################### Columns #####################")
    print(dataframe.columns)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("################### Describe ###################")
    print(dataframe.describe().T)

check_df(df)

# Bazı columns isimleri küçük harfle ve ' education' gibi başında boşluk olacak şekilde yazılmış onları düzeltiyoruz.
df.columns = df.columns.str.replace(' ', '')
df.columns = [col.upper() for col in df.columns]

# loan_status değişkenini approved(1), rejected(0) haline getiriyoruz.
df["LOAN_STATUS"] = df["LOAN_STATUS"].apply(lambda  x: 1 if x == " Approved" else 0)

# Verinin değerleri genel olarak incelendiğinde ilk dikkat çeken şey residential_assets_value değişkeninde bulunan
# eksik değerler. Bu değerleri 0 yapıyoruz.
num = df._get_numeric_data()
num[num < 0 ] = 0

df.sort_values("RESIDENTIAL_ASSETS_VALUE", ascending=True)

df.head()



########################## NUMERİK, KATEGORİK DEĞİŞKENLERİN YAKALANMASI VE İNCELENMESİ #################################

# Kategorik, numerik, cat_but_car olan değişkenlerin analizini yapan fonksiyon;
def grab_col_names(dataframe, cat_th=5, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Kategorik değişkenlerin sayısını ve oranlarını veren fonksiyon;
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(), "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df,col)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    if plot:
        dataframe[numerical_col].hist(bins=50)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)
    print("#####################################")

for col in num_cols:
    num_summary(df, col, True)

# numerik değişkenlerin target a göre analizi
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "LOAN_STATUS", col)


# kategorik değişkenlerin target a göre analizi
def target_summary_with_cat(dataframe, target, cat_cols):
    print(pd.DataFrame({ "COUNT": dataframe.groupby(cat_cols)[target].value_counts(),
                         "RATIO": dataframe.groupby(cat_cols)[target].value_counts() / len(dataframe)}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df,"LOAN_STATUS",col)



########################################### KORELASYON ANALİZİ #########################################################

df[num_cols].corr()

# Korelasyon Matrisi
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show(block=True)

########################################### FEATURE ENGINEERING ########################################################
# Eksik değer kontrolü
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

missing_values_table(df)


# Aykırı değerlerin IQR yöntemi ile baskılanması
def outlier_thresholds(dataframe, variable, low_quantile=0.10, up_quantile=0.90):
    quantile_one = dataframe[variable].quantile(low_quantile)
    quantile_three = dataframe[variable].quantile(up_quantile)
    interquantile_range = quantile_three - quantile_one
    up_limit = quantile_three + 1.5 * interquantile_range
    low_limit = quantile_one - 1.5 * interquantile_range
    return low_limit, up_limit

# Aykırı değer kontrolü
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    if col != "LOAN_STATUS":
      print(col, check_outlier(df, col))



######################################### BASE MODEL KURULUMU ##########################################################
dff = df.copy()
cat_cols = [col for col in cat_cols if col not in ["LOAN_STATUS"]]
cat_cols

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
dff = one_hot_encoder(dff, cat_cols, drop_first=True)

# train ve test verisinin ayrılması.

y = dff["LOAN_STATUS"]
X = dff.drop(["LOAN_STATUS", "LOAN_ID"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=23)


models = [('LR', LogisticRegression(random_state=23)),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier(random_state=23)),
          ('RF', RandomForestClassifier(random_state=23)),
          ('XGB', XGBClassifier(random_state=23)),
          ("LightGBM", LGBMClassifier(random_state=23)),
          ("CatBoost", CatBoostClassifier(verbose=False, random_state=23))]


for name, model in models:
    cv_results = cross_validate(model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")



############################################# ÖZELLİK ÇIKARIMI #########################################################

# Taşınmaz varlıkların değeri
df["Long_Term_Assets"] = df["RESIDENTIAL_ASSETS_VALUE"]

# Taşınabilir varkıların değeri
df["Current_Assets"] = df["COMMERCIAL_ASSETS_VALUE"] + df["LUXURY_ASSETS_VALUE"] +df["BANK_ASSET_VALUE"]

# Toplam varlıkların değeri
df["Total_Assets"] = df["Long_Term_Assets"] + df["Current_Assets"]

df.drop(columns=['RESIDENTIAL_ASSETS_VALUE', 'COMMERCIAL_ASSETS_VALUE', 'LUXURY_ASSETS_VALUE', 'BANK_ASSET_VALUE'], inplace=True)

df["Monthly_loan_payment"] = (df["LOAN_AMOUNT"] / (df["LOAN_TERM"]*12)).astype(int)

df.drop(columns=['LOAN_AMOUNT', 'LOAN_TERM'], inplace=True)

# Kredi skoru değişkenini (CIBIL_SCORE) değişkeninin kategorilere ayırılıp yeni risk segmenti değişkeni oluşturulması
df["C_Score_Risk_Segm"] = pd.qcut(df['CIBIL_SCORE'], q= 5, labels=["most_risky", "medium_risk", "low_risk", "good", "very_good"])

df.head()
df.shape


############################################## ENCODING ################################################################
cat_cols, cat_but_car, num_cols = grab_col_names(df)
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]
binary_cols

for col in binary_cols:
    df = label_encoder(df, col)

df.head()
df.shape

num_cols = ['NO_OF_DEPENDENTS', 'INCOME_ANNUM', 'CIBIL_SCORE', 'Long_Term_Assets', 'Current_Assets', 'Total_Assets', 'Monthly_loan_payment']



############################################# MODELLEME ################################################################
# train ve test verisinin ayrılması.
y = df["LOAN_STATUS"]
X = df.drop(["LOAN_STATUS", "LOAN_ID", "C_Score_Risk_Segm"], axis=1)

scaler = StandardScaler()

X[num_cols] = StandardScaler().fit_transform(X[num_cols])

X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.20, random_state=23)

def plot_confusion_matrix(y_test, y_pred):
    acc = round(accuracy_score(y_test, y_pred), 2)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f", cmap ='Blues')
    plt.xlabel('Prediction')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


####### XGBOOST

xgboost_model = XGBClassifier(random_state=23)
xgboost_model.get_params()

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [2, 3],
                  "n_estimators": [50, 100],
                  "colsample_bytree": [0.5, 0.7]}

xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X_train, y_train)
xgboost_best_grid.best_params_

xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_, random_state=23).fit(X_train, y_train)

y_pred = xgboost_final.predict(X_test)

# Confusion Matrix
plot_confusion_matrix(y_test, y_pred)

# Değerlendirme Raporu
print(classification_report(y_test, y_pred, digits=3))

# Accuracy : 0.956
# Recall : 0.982
# Precision : 0.945
# F1_Score : 0.963



###### LIGHTGBM

lgbm_model = LGBMClassifier(random_state=23)
lgbm_model.get_params()

lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [50, 100, 200],
               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X_train, y_train)
lgbm_best_grid.best_params_

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=23).fit(X_train, y_train)

y_pred = lgbm_final.predict(X_test)

# Confusion Matrix
plot_confusion_matrix(y_test, y_pred)

# Değerlendirme Raporu
print(classification_report(y_test, y_pred, digits=3))

# Accuracy : 0.971
# Recall : 0.968
# Precision : 0.985
# F1_Score : 0.976



####### CATBOOST

catboost_model = CatBoostClassifier(random_state=23, verbose=False)
catboost_model.get_params()

catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}

catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=False).fit(X_train, y_train)
catboost_best_grid.best_params_

catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_, random_state=23).fit(X_train, y_train)

y_pred = catboost_final.predict(X_test)

# Confusion Matrix
plot_confusion_matrix(y_test, y_pred)

# Değerlendirme Raporu
print(classification_report(y_test, y_pred, digits=3))

# Accuracy : 0.966
# Recall : 0.964
# Precision : 0.981
# F1_Score : 0.972



###### RANDOM FOREST

rf_model = RandomForestClassifier(random_state=23)
rf_model.get_params()

rf_params = {"max_depth": [5, None],
             "max_features": [5, 7, "sqrt"],
             "min_samples_split": [2, 5, 10],
             "n_estimators": [50, 100, 200]}

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=1).fit(X_train, y_train)

rf_best_grid.best_params_

rf_final = RandomForestClassifier().set_params(**rf_best_grid.best_params_).fit(X_train, y_train)

y_pred = rf_final.predict(X_test)

# Confusion Matrix
plot_confusion_matrix(y_test, y_pred)

# Değerlendirme Raporu
print(classification_report(y_test, y_pred, digits=3))

# Accuracy : 0.971
# Recall : 0.972
# Precision : 0.981
# F1_Score : 0.976

# Modeli dosyaya yazma
joblib.dump(rf_final, 'rf_model.joblib')



# FEATURE IMPORTANCE
def plot_importance(model, features, num=len(X), save=False):

    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")


plot_importance(rf_final, X, num=30)

